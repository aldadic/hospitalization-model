from dash import Dash, html, dcc, ctx
from src.utils import load_config
from src.data_loader import CaseData, OccupancyData
from src.causal_model import CausalModel
from src.arima_model import ArimaModel
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
           title='Hospitalization Model', update_title=None)
server = app.server

CONFIG = load_config("config/general.yaml", "config/causal_model.yaml", "config/confidence.yaml")
CASE_DATA = CaseData("data/age_groups.csv")
OCCUPANCY_DATA = OccupancyData("data/hospitalization.csv")
causal_model = CausalModel(
    hospitalization_p=CONFIG["causal_model"]["params"]["hospitalization_p"],
    poisson_lambda=CONFIG["causal_model"]["params"]["poisson_lambda"],
    truncnorm_loc=CONFIG["causal_model"]["params"]["truncnorm_loc"],
    truncnorm_scale=CONFIG["causal_model"]["params"]["truncnorm_scale"],
    state=CONFIG["general"]["state"],
    bed_type=CONFIG["general"]["type"],
    age_groups=tuple(CONFIG["general"]["age_groups"]),
    from_date=pd.to_datetime(CONFIG["general"]["date"]) - pd.Timedelta(days=30),
    to_date=CONFIG["general"]["date"],
    buffer=CONFIG["causal_model"]["buffer"],
    case_data=CASE_DATA,
    occupancy_data=OCCUPANCY_DATA
)
arima_model = ArimaModel(
    state=CONFIG["general"]["state"],
    bed_type=CONFIG["general"]["type"],
    age_groups=CONFIG["general"]["age_groups"],
    from_date=OCCUPANCY_DATA.min_date(),
    to_date=CONFIG["general"]["date"],
    case_data=CASE_DATA,
    occupancy_data=OCCUPANCY_DATA
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checkbox(
                                id="plot-causal-model",
                                label="Causal Model",
                                value=True,
                                label_class_name="mb-0",
                                label_id="plot-causal-model-target",
                                label_style={"textDecoration": "underline", "cursor": "pointer"}
                            ),
                            dbc.Tooltip(
                                ("Simulated using the parameters below. "
                                 "Prediction interval is only meaningful if interval = 6 days."),
                                target="plot-causal-model-target",
                                placement="bottom"
                            ),
                        ], width=3),
                        dbc.Col([
                            dbc.Checkbox(
                                id="plot-arima-model",
                                label="ARIMA Model",
                                value=False,
                                label_class_name="mb-0",
                                label_id="plot-arima-model-target",
                                label_style={"textDecoration": "underline", "cursor": "pointer"}
                            ),
                            dbc.Tooltip(
                                ("ARIMA model is always calibrated with auto_arima using all available data, "
                                 "which is why this takes a while to load."),
                                target="plot-arima-model-target",
                                placement="bottom"
                            ),
                        ], width=3),
                        dbc.Col([
                            dbc.Button('Plot result', id='draw', n_clicks=0, className='w-100')
                        ], width=3),
                    ], justify='evenly', class_name='align-items-center'),
                ])
            ])
        ], width=6)
    ], justify='center'),
    dbc.Spinner(dcc.Graph(
        id='figure'
    ), spinner_style={"width": "5rem", "height": "5rem"}),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("General Settings"),
                dbc.CardBody([
                    dbc.Row([
                        html.Div([
                            dbc.Label('Forecast from: ', html_for="date-picker")
                        ], className='col-5'),
                        html.Div([
                            dcc.DatePickerSingle(
                                id='date-picker',
                                min_date_allowed=OCCUPANCY_DATA.min_date(),
                                max_date_allowed=OCCUPANCY_DATA.max_date(),
                                date=pd.to_datetime(CONFIG["general"]["date"]),
                                display_format='DD.MM.YYYY'
                            )
                        ], className='col-7')
                    ], className='mb-2 align-items-center'),
                    dbc.Row([
                        html.Div([
                            dbc.Label('Forecast days: ', html_for="date-picker")
                        ], className='col-5'),
                        html.Div([
                            dbc.Input(id='forecast_days-input', value=CONFIG["general"]["forecast_days"],
                                      type='number', step=1)
                        ], className='col-7')
                    ], className='mb-2 align-items-center'),
                    dbc.Row([
                        html.Div([
                            dbc.Label('Federal State: ', html_for="state-select")
                        ], className='col-5'),
                        html.Div([
                            dcc.Dropdown(
                                id='state-select',
                                value=CONFIG["general"]["state"],
                                options=[{"label": state, "value": state} for state in CASE_DATA.states()],
                                clearable=False
                            )
                        ], className='col-7')
                    ], className='mb-2 align-items-center'),
                    dbc.Row([
                        html.Div([
                            dbc.Label('Type: ', html_for="type-select"),
                        ], className='col-5'),
                        html.Div([
                            dcc.Dropdown(
                                id='type-select',
                                value=CONFIG["general"]["type"],
                                options=[
                                    {"label": "ICU", "value": "ICU"},
                                    {"label": "normal", "value": "normal"}
                                ],
                                clearable=False
                            )
                        ], className='col-7')
                    ], className='mb-2 align-items-center'),
                    html.Div([
                        dbc.Label('Age Groups: ', html_for="age-select"),
                        dcc.Dropdown(
                            id='age-select',
                            value=CONFIG["general"]["age_groups"],
                            options=[{"label": age_group, "value": age_group} for age_group in CASE_DATA.age_groups()],
                            multi=True
                        )
                    ], className='mb-2')
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Causal Model - Settings"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Simulation options: '),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Interval:"),
                                    dbc.Input(id='interval-input', value=30, type='number',
                                              step=1, class_name="text-end"),
                                    dbc.InputGroupText("days"),
                                ],
                                className="mb-2",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Buffer:"),
                                    dbc.Input(id='buffer-input', value=CONFIG["causal_model"]["buffer"], type='number',
                                              step=1, class_name="text-end"),
                                    dbc.InputGroupText("days"),
                                ],
                                className="mb-2",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Monte Carlo iterations:"),
                                    dbc.Input(id='monte-carlo-input', type='number', step=1, class_name="text-end",
                                              value=CONFIG["causal_model"]["monte_carlo_iterations"])
                                ],
                                className="mb-2",
                            ),
                            dbc.Label('Calibration options: '),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Max. steps:"),
                                    dbc.Input(id='max-iter-input', type='number', step=1, class_name="text-end",
                                              value=CONFIG["causal_model"]["max_iter"])
                                ],
                                className="mb-2",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.Checkbox(
                                        id="use-x0",
                                        label="Use current parameters as initial value in calibration",
                                        value=False,
                                    ),
                                ]
                            )
                        ], width=4),
                        dbc.Col([
                            dbc.Label('hospitalization_p range: ', html_for="hospitalization_p-range-input"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id='p_min',
                                              value=CONFIG["causal_model"]["param_ranges"]["hospitalization_p"][0]),
                                    dbc.InputGroupText("to"),
                                    dbc.Input(id='p_max', class_name="text-end",
                                              value=CONFIG["causal_model"]["param_ranges"]["hospitalization_p"][1]),
                                ],
                                id="hospitalization_p-range-input", className="mb-2",
                            ),
                            dbc.Label('poisson_lambda range: ', html_for="poisson_lambda-range-input"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id='poisson_lambda_min',
                                              value=CONFIG["causal_model"]["param_ranges"]["poisson_lambda"][0]),
                                    dbc.InputGroupText("to"),
                                    dbc.Input(id='poisson_lambda_max',
                                              value=CONFIG["causal_model"]["param_ranges"]["poisson_lambda"][1],
                                              class_name="text-end"),
                                ],
                                id="poisson_lambda-range-input", className="mb-2",
                            ),
                            dbc.Label('truncnorm_loc range: ', html_for="truncnorm_loc-range-input"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id='truncnorm_loc_min',
                                              value=CONFIG["causal_model"]["param_ranges"]["truncnorm_loc"][0]),
                                    dbc.InputGroupText("to"),
                                    dbc.Input(id='truncnorm_loc_max', class_name="text-end",
                                              value=CONFIG["causal_model"]["param_ranges"]["truncnorm_loc"][1]),
                                ],
                                id="truncnorm_loc-range-input", className="mb-2",
                            ),
                            dbc.Label('truncnorm_scale range: ', html_for="truncnorm_scale-range-input"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id='truncnorm_scale_min',
                                              value=CONFIG["causal_model"]["param_ranges"]["truncnorm_scale"][0]),
                                    dbc.InputGroupText("to"),
                                    dbc.Input(id='truncnorm_scale_max', class_name="text-end",
                                              value=CONFIG["causal_model"]["param_ranges"]["truncnorm_scale"][1]),
                                ],
                                id="truncnorm_scale-range-input", className="mb-2",
                            )
                        ], width=4),
                        dbc.Col([
                            dbc.Label('Parameters: '),
                            dbc.Spinner([dbc.InputGroup(
                                [
                                    dbc.InputGroupText(f"{key}:"),
                                    dbc.Input(id=f"{key}", value=value, class_name="text-end"),
                                ],
                                className="mb-2"
                            ) for key, value in CONFIG["causal_model"]["params"].items()]),
                            dbc.Button('Calibrate', id='calibrate', n_clicks=0, className='w-100', color='primary')
                        ])
                    ])
                ])
            ]),
        ], width=8)
    ], className="g-3 pt-3 justify-content-center"),
    dcc.ConfirmDialog(
        id='confirm-calibration'
    ),
], class_name="pt-4")


@app.callback(
    Output('hospitalization_p', 'value'),
    Output('poisson_lambda', 'value'),
    Output('truncnorm_loc', 'value'),
    Output('truncnorm_scale', 'value'),
    Output('confirm-calibration', 'displayed'),
    Output('confirm-calibration', 'message'),
    Input('calibrate', 'n_clicks'),
    State('date-picker', 'date'),
    State('state-select', 'value'),
    State('age-select', 'value'),
    State('type-select', 'value'),
    State('p_min', 'value'),
    State('p_max', 'value'),
    State('poisson_lambda_min', 'value'),
    State('poisson_lambda_max', 'value'),
    State('truncnorm_loc_min', 'value'),
    State('truncnorm_loc_max', 'value'),
    State('truncnorm_scale_min', 'value'),
    State('truncnorm_scale_max', 'value'),
    State('interval-input', 'value'),
    State('buffer-input', 'value'),
    State('monte-carlo-input', 'value'),
    State('max-iter-input', 'value'),
    State('use-x0', 'value'),
    State('hospitalization_p', 'value'),
    State('poisson_lambda', 'value'),
    State('truncnorm_loc', 'value'),
    State('truncnorm_scale', 'value')
)
def calibration(button_calibrate, date, state, age_groups, bed_type, p_min, p_max,
                poisson_lambda_min, poisson_lambda_max, truncnorm_loc_min, truncnorm_loc_max, truncnorm_scale_min,
                truncnorm_scale_max, interval, buffer, n, maxiter, use_x0, hospitalization_p, poisson_lambda,
                truncnorm_loc, truncnorm_scale):

    # don't do anything if the button hasn't been clicked
    if ctx.triggered[0]["prop_id"] != 'calibrate.n_clicks':
        return CONFIG["causal_model"]["params"]["hospitalization_p"], \
               CONFIG["causal_model"]["params"]["poisson_lambda"], \
               CONFIG["causal_model"]["params"]["truncnorm_loc"], \
               CONFIG["causal_model"]["params"]["truncnorm_scale"], False, ''

    causal_model.hospitalization_p = float(hospitalization_p)
    causal_model.poisson_lambda = float(poisson_lambda)
    causal_model.truncnorm_loc = float(truncnorm_loc)
    causal_model.truncnorm_scale = float(truncnorm_scale)
    to_date = pd.to_datetime(date)
    from_date = to_date - pd.Timedelta(days=interval)
    causal_model.update(state, bed_type, age_groups, from_date, to_date, buffer)

    bounds = [
        (float(p_min), float(p_max)),
        (float(poisson_lambda_min), float(poisson_lambda_max)),
        (float(truncnorm_loc_min), float(truncnorm_loc_max)),
        (float(truncnorm_scale_min), float(truncnorm_scale_max))
    ]

    result = causal_model.calibrate(n, bounds, maxiter, use_x0, disp=True)
    message = f"Calibration completed. MAPE: {result.fun:.5f}"

    return result.x[0], result.x[1], result.x[2], result.x[3], True, message


@app.callback(
    Output('figure', 'figure'),
    Input('draw', 'n_clicks'),
    State('plot-causal-model', 'value'),
    State('plot-arima-model', 'value'),
    State('date-picker', 'date'),
    State('state-select', 'value'),
    State('age-select', 'value'),
    State('type-select', 'value'),
    State('forecast_days-input', 'value'),
    State('interval-input', 'value'),
    State('buffer-input', 'value'),
    State('monte-carlo-input', 'value'),
    State('hospitalization_p', 'value'),
    State('poisson_lambda', 'value'),
    State('truncnorm_loc', 'value'),
    State('truncnorm_scale', 'value')
)
def draw_figure(button_draw, plot_causal, plot_arima, date, state, age_groups, bed_type, forecast_days,
                interval, buffer, n, hospitalization_p, poisson_lambda, truncnorm_loc, truncnorm_scale):

    # only draw if the button has been clicked
    if ctx.triggered[0]["prop_id"] != 'draw.n_clicks':
        return {}

    date = pd.to_datetime(date)
    start = date - pd.Timedelta(days=max(interval, 30))
    end = date + pd.Timedelta(days=forecast_days)
    layout = go.Layout(
        margin=go.layout.Margin(l=0, r=0, b=30, t=30)
    )
    fig = go.Figure(layout=layout)
    reference = OCCUPANCY_DATA.get_df(start, end, state, bed_type)
    fig.add_vrect(x0=date+pd.Timedelta(days=1), x1=end, annotation_text="forecast",
                  annotation_position="top left", fillcolor="gray", opacity=0.15, line_width=0)
    fig.update_layout(xaxis_range=[start, end], showlegend=True)
    lines = [
        go.Scatter(x=reference.index, y=reference["occupancy"], mode='lines+markers',
                   name='Actual', line=dict(color='#ef553b'))
    ]
    areas = []

    if plot_causal:
        start = date - pd.Timedelta(days=interval)
        causal_model.hospitalization_p = float(hospitalization_p)
        causal_model.poisson_lambda = float(poisson_lambda)
        causal_model.truncnorm_loc = float(truncnorm_loc)
        causal_model.truncnorm_scale = float(truncnorm_scale)
        causal_model.update(state, bed_type, age_groups, start, end, buffer)
        simulation = causal_model.simulate(n)
        x_axis = pd.date_range(start=start, end=end, freq="D")
        lines.append(go.Scatter(x=x_axis, y=simulation, mode='lines+markers', name='Causal Model',
                                line=dict(color='#636efa')))
        quantiles = pd.DataFrame(CONFIG["quantiles"]["causal_model"], columns=["lower", "upper"])
        lower = list(simulation[-14:] + quantiles["lower"] * simulation[-14:])
        upper = list(simulation[-14:] + quantiles["upper"] * simulation[-14:])
        x = list(x_axis[-14:])
        areas.append(go.Scatter(x=x+x[::-1], y=upper+lower[::-1], fill='toself',
                     fillcolor='rgba(99, 110, 250, 0.2)', line_color='rgba(99, 110, 250, 0.2)',
                     name='Causal Model Prediction Interval'))

    if plot_arima:
        arima_model.to_date = date
        arima_model.calibrate()
        pred, conf_int = arima_model.predict(forecast_days)
        x_axis = list(pd.date_range(start=date, end=end, freq="D")[1:])
        lines.append(go.Scatter(x=x_axis, y=pred, mode='lines+markers', name='ARIMA Model', line=dict(color='#ab63fa')))
        upper = list(conf_int[:, 1])
        lower = list(conf_int[:, 0])
        areas.append(go.Scatter(x=x_axis+x_axis[::-1], y=upper+lower[::-1], fill='toself',
                                fillcolor='rgba(171, 99, 250, 0.2)', line_color='rgba(171, 99, 250, 0.2)',
                                name='ARIMA Confidence Interval'))
        quantiles = pd.DataFrame(CONFIG["quantiles"]["arima_model"], columns=["lower", "upper"])
        lower = list(pred + quantiles["lower"] * pred)
        upper = list(pred + quantiles["upper"] * pred)
        areas.append(go.Scatter(x=x_axis+x_axis[::-1], y=upper+lower[::-1], fill='toself',
                                fillcolor='rgba(255, 102, 146, 0.2)', line_color='rgba(255, 102, 146, 0.2)',
                                name='ARIMA Prediction Interval'))

    fig.add_traces(areas)
    fig.add_traces(lines)

    return fig


if __name__ == '__main__':
    app.run(port="8050", debug=False)
