import numpy as np
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objs as pgo
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, MATCH,ALL, callback_context
import dash_bootstrap_components as dbc
import os
import ast


CSV_LOG_PATH = "resultados_algoritmos.csv"



if not os.path.exists(CSV_LOG_PATH):
    import csv
    with open(CSV_LOG_PATH, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Dataset", "Algoritmo", "Parametros", "Aciertos (%)"])
        writer.writeheader()




parametros_por_algoritmo = {
    "DBscan": ["eps", "n_min"],
    "HDBscan": ["n_min"],
    "Density peak":["d_percent","n_clusters"],
    "Quickshift":["sigma","n_min"],
    "otro_algo": ["rango", "umbral"]
}
datos_formas = pd.read_csv("datos.txt")
datos = datos_formas.copy()
data_compound=pd.read_csv("compound.txt")
pathbased=pd.read_csv("pathbased_1")
clust_algorithms=['none','DBscan','HDBscan','Density peak','Quickshift']
datasets={
    'Pathbased':pathbased,
    'Formas': datos,
    'Compound':data_compound

}
general_table=html.Div(id="tabla-resultados")
algorithm_representation = html.Div([
    html.H3(html.H2("Visualizador de Datasets"),

                        # Selector de dataset

                        "Selecciona los datos y el metodo de clusterizacion"),
                        dcc.Dropdown(
                                id='data_dropdown',
                                options=[{'label': k, 'value': k} for k in datasets.keys()],
                                value=list(datasets.keys())[0]
                            ),
                        html.Div(id='dataset_selected'),
                        html.Label("Selecciona el algoritmo de clustering"),
                        dcc.Dropdown(
                            id='algorithm_dropdown',
                            options=[{'label': k, 'value': k} for k in clust_algorithms],
                            value='none'  # valor inicial, será actualizado dinámicamente
                        ),
                        dcc.Store(id='store-dbscan-parametros'),
                        dcc.Store(id='clustered_data'),
                        dbc.Button("Abrir/cerrar parámetros algoritmos", id="btn-popover", n_clicks=0),
                        dbc.Popover(
                                [
                                    dbc.PopoverHeader("Parámetros algoritmos"),
                                    dbc.PopoverBody([
                                        dbc.PopoverBody(id="popover-body"),  # ← Este es dinámico

                                    ])
                                ],
                                id="popover-dbscan",
                                target="btn-popover",  # Ancla el popover al botón
                               # trigger="click",       # Se abre al hacer clic
                                placement="bottom",    # Aparece debajo del botón
                                is_open=False,          # Estado inicial cerrado
                                style={
                                        "backgroundColor": "white",
                                        "padding": "15px",
                                        "borderRadius": "10px",
                                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)",
                                        "zIndex": 2000  # Asegura que se superponga a lo de detrás
                                    }
                            ),
                       html.Hr(),
                       #dcc.Graph(id="scatter-plot"),
                        html.Div([
                            html.Div([
                                dcc.Graph(id='scatter-plot-solution')
                            ], style={"width": "50%", "display": "inline-block"}),

                            html.Div([
                                dcc.Graph(id='scatter-plot-algorithm')
                            ], style={"width": "50%", "display": "inline-block"}),
                            html.Div(html.Button("Guardar resultado", id="btn-guardar", n_clicks=0, className="btn btn-success"),)

                        ], style={"display": "flex"}),])
app = dash.Dash(__name__,suppress_callback_exceptions=True)

# Si no existe la columna "grupo_manual", créala
if "grupo_DBscan" not in datos.columns:
    coords=datos[['x','y']]
    dbscan = DBSCAN(eps=1.1, min_samples=5)
    datos["grupo_DBscan"] =dbscan.fit_predict(coords)
if "grupo_manual" not in datos.columns:
    datos["grupo_manual"] = datos["group"]  # Inicialmente, es igual a "group"
datos['id'] = datos.index
todos_los_grupos = pd.DataFrame({"group": datos["group"].astype(str).unique()})

def get_icon(icon):
    return DashIconify(icon=icon, height=16)


app.layout = dmc.MantineProvider(
    html.Div([
        dcc.Location(id="_pages_location"),
        dcc.Store(id="store_sidebar_visible", data=True),
        dcc.Store(id="store_guardado"),  # ← Añadido


        # Botón superior

        dmc.Group(
            children=[
                dmc.Button("Mostrar/Ocultar menú", id="toggle_sidebar", variant="light")
            ],
            style={"marginTop": "1rem", "marginBottom": "1rem", "justifyContent": "flex-start"}
        ),

        # Contenedor flex verdadero
        html.Div([
            # Menú lateral
            dmc.Stack(
                id="sidebar",
                children=[
                    dmc.NavLink(label="Representación algoritmos", href="/algorithm_use", active="partial",
                                id={"type": "navlink", "index": "/algorithm_use"}),
                    dmc.NavLink(label="Representación gráfica",  children=[
                        dmc.NavLink(label="Comparación de algoritmos", children=[
                            dmc.NavLink(label="Dataset: datos formas", href="/comparison_datos_formas", active="partial",
                                id={"type": "navlink", "index": "/comparison_datos_formas"}),
                            dmc.NavLink(label="Dataset: Compound", href="/comparison_compound", active="partial",
                                id={"type": "navlink", "index": "/comparison_compound"}),
                            dmc.NavLink(label="Dataset: Pathbased", href="/comparison_pathbased", active="partial",
                                id={"type": "navlink", "index": "/comparison_pathbased"}),

                        ]),
                        dmc.NavLink(label="Puntos de dispersión"),
                        dmc.NavLink(label="Grafico de barras"),
                    ]),
                    dmc.NavLink(label="Tabla de resultados", children=[
                        dmc.NavLink(label="Tabla general", href="/general_table", active="partial",
                                id={"type": "navlink", "index": "/general_table"}),

                        dmc.NavLink(label="Tabla de datos con distintos algoritmos"),
                        dmc.NavLink(label="Tabla de algoritmos con distintos datos"),
                    ]),

                ],
                style={
                    "width": "20%",
                    "height": "100vh",
                    "borderRight": "1px solid #ccc",
                    "padding": "1rem"
                }
            ),

            # Contenido principal
            html.Div(
                id="page-content",
                style={
                    "width": "80%",
                    "padding": "2rem"
                }
            )
        ], style={"display": "flex", "width": "100%"})  # ← Aquí está la clave
    ])
)

@app.callback(
    Output({"type": "navlink", "index": ALL}, "active"),
    Input("_pages_location", "pathname")
)
def update_navlinks(pathname):
    return [
        "partial" if pathname.startswith(control["id"]["index"]) else False
        for control in callback_context.outputs_list
    ]
@app.callback(
    Output("store_sidebar_visible", "data"),
    Input("toggle_sidebar", "n_clicks"),
    State("store_sidebar_visible", "data"),
    prevent_initial_call=True
)
def toggle_sidebar(n, current_state):
    return not current_state
@app.callback(
    Output("sidebar", "style"),
    Output("page-content", "style"),
    Input("store_sidebar_visible", "data")
)
def ajustar_estilos_menu(visible):
    if visible:
        return (
            {
                "width": "20%",
                "height": "100vh",
                "borderRight": "1px solid #ccc",
                "padding": "1rem"
            },
            {
                "width": "80%",
                "padding": "2rem"
            }
        )
    else:
        return (
            {"display": "none"},
            {
                "width": "100%",
                "padding": "2rem"
            }
        )
def higher_percentage_parameters(data_name,df_log,algorithm,default_parameters):

        df_log["Aciertos (%)"] = pd.to_numeric(df_log["Aciertos (%)"], errors="coerce")

        # Filtrar por dataset y algoritmo
        filter = (df_log["Dataset"] == data_name) & (df_log["Algoritmo"] == algorithm)
        df_filtered = df_log[filter]

        if df_filtered.empty:
            return default_parameters

        best_row = df_filtered.loc[df_filtered["Aciertos (%)"].idxmax()]
        parametros = best_row["Parametros"]
        if isinstance(parametros, str):
            parametros = ast.literal_eval(parametros)

        return parametros

def Density_peak(df,parameters):
    X = df[['x', 'y']].values
    distances = pairwise_distances(X)
    dc = np.percentile(distances, parameters["d_percent"])
    rho = np.sum(distances < dc, axis=1) - 1
    delta = np.zeros_like(rho, dtype=float)
    nearest_higher = np.zeros_like(rho, dtype=int)

    for i in range(len(X)):
        higher = np.where(rho > rho[i])[0]
        if len(higher) > 0:
            j = higher[np.argmin(distances[i, higher])]
            delta[i] = distances[i, j]
            nearest_higher[i] = j
        else:
            delta[i] = np.max(distances[i])
            nearest_higher[i] = i
        # Paso 3: Calcular gamma
    gamma = rho * delta
    centers = np.argsort(gamma)[-parameters["n_clusters"]:]

    # Paso 4: Asignar clústeres
    labels = -np.ones(len(X), dtype=int)
    for idx, c in enumerate(centers):
        labels[c] = idx

    order = np.argsort(-rho)
    for i in order:
        if labels[i] == -1:
            labels[i] = labels[nearest_higher[i]]
    return labels
def Quickshift(df,parameters):
    X = df[['x', 'y']].values
    n = len(X)
    influencia=parameters["sigma"]
    # Estimación de densidad por kernel Gaussiano
    nbrs = NearestNeighbors(radius=3 * influencia).fit(X)
    radius_neighbors = nbrs.radius_neighbors(X, return_distance=True)

    densities = np.zeros(n)
    for i in range(n):
        distances_i = radius_neighbors[0][i]
        densities[i] = np.sum(np.exp(- (distances_i ** 2) / (2 * influencia ** 2)))

    # Conexión a vecino más cercano con mayor densidad
    parents = np.full(n, -1)
    for i in range(n):
        neighbors = radius_neighbors[1][i]
        valid = [j for j in neighbors if densities[j] > densities[i]]
        if valid:
            j_min = valid[np.argmin(np.linalg.norm(X[valid] - X[i], axis=1))]
            parents[i] = j_min
        else:
            parents[i] = i  # raíz

    # Asignar etiquetas de clúster a cada raíz distinta
    cluster_map = {}
    labels = np.full(n, -1)
    cluster_id = 0

    for i in range(n):
        # Seguir el árbol hasta la raíz
        path = []
        node = i
        while parents[node] != node:
            path.append(node)
            node = parents[node]
        root = node

        # Crear nuevo clúster si no existe
        if root not in cluster_map:
            cluster_members = [j for j in range(n) if parents[j] == root or j == root]
            if len(cluster_members) >= parameters["n_min"]:
                cluster_map[root] = cluster_id
                cluster_id += 1
            else:
                cluster_map[root] = -1  # ruido

        labels[i] = cluster_map[root]
    return labels
def comparison_data(df,data_name,CSV):
    df_log = pd.read_csv(CSV, encoding="latin1")
    #data=datasets[df]
    coords = df[['x', 'y']]

    figure_solution = px.scatter(df, x="x", y="y", color="group",
                                 hover_data=["group"])
    dbscan_param={"eps": 1.1, "n_min": 5}
    dbscan_param=higher_percentage_parameters(data_name,df_log,"DBscan",dbscan_param)

    if dbscan_param is None:
        dbscan_param = {"eps": 1.1, "n_min": 5}
    dbscan = DBSCAN(eps=dbscan_param["eps"], min_samples=dbscan_param["n_min"])
    df["DBscan"] = dbscan.fit_predict(coords)
    figure_DBscan = px.scatter(df, x="x", y="y", color="DBscan",hover_data=["group", "DBscan"])

    df = df.sort_values(by=["x", "y"]).reset_index(drop=True)
    hdbscan_param={"n_min": 5}
    hdbscan_param=higher_percentage_parameters(data_name,df_log,"HDBscan",hdbscan_param)


    hdbscan_a = hdbscan.HDBSCAN(min_cluster_size=hdbscan_param["n_min"])
    df["HDBscan"] = hdbscan_a.fit_predict(df)
    figure_HDBscan = px.scatter(df, x="x", y="y", color="HDBscan",hover_data=["group", "HDBscan"])

    df = df.sort_values(by=["x", "y"]).reset_index(drop=True)
    densitypeak_param={"d_percent": 1, "n_clusters": 5}
    densitypeak_param=higher_percentage_parameters(data_name,df_log,"Density Peak",densitypeak_param)


    df["Density peak"] = Density_peak(df,densitypeak_param)
    figure_densitypeak = px.scatter(df, x="x", y="y", color="Density peak",hover_data=["group", "Density peak"])

    df = df.sort_values(by=["x", "y"]).reset_index(drop=True)
    quickshift_param={"sigma": 1, "n_min": 5}
    quickshift_param=higher_percentage_parameters(data_name,df_log,"Quickshift",quickshift_param)


    df["Quickshift"] = Quickshift(df,quickshift_param)
    figure_quickshift = px.scatter(df, x="x", y="y", color="Quickshift",hover_data=["group", "Quickshift"])
    return [figure_DBscan,figure_HDBscan,figure_densitypeak,figure_quickshift]


def figures_comparison(figures):
    return dmc.Paper(
    children=[
        dmc.Title("Resultados de clustering", order=2, mb="md"),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dmc.Text("Resultado de DBSCAN", size="sm", mt="xs"),
                        dcc.Graph(figure=figures[0]),  # DBSCAN
                    ],
                        style={"width": "50%", "display": "inline-block"}
                ),
                html.Div(
                    children=[
                        dmc.Text("Resultado de HDBSCAN", size="sm", mt="xs"),

                        dcc.Graph(figure=figures[1]),  # HDBSCAN
                    ],
                    style={"width": "50%", "display": "inline-block"}
                ),
                html.Div(
                    children=[
                        dmc.Text("Resultado de Density Peak", size="sm", mt="xs"),
                        dcc.Graph(figure=figures[2]),  # Density Peak
                    ],
                    style={"width": "50%", "display": "inline-block"}
                ),
                html.Div(
                    children=[
                        dmc.Text("Resultado de Quickshift", size="sm", mt="xs"),
                        dcc.Graph(figure=figures[3]),  # Quickshift
                    ],
                    style={"width": "50%", "display": "inline-block"}
                ),
            ],

        )
    ],
    shadow="sm",
    radius="md",
    p="md",
    withBorder=True
)

def table_page(CSV):
    df_log = pd.read_csv(CSV, encoding="latin1")
    df_log["Aciertos (%)"] = pd.to_numeric(df_log["Aciertos (%)"], errors="coerce")
    df_log.sort_values(by=["Dataset", "Algoritmo", "Aciertos (%)"], ascending=[True, True, False], inplace=True)
    df_log["Destacado"] = False
    idx_max = df_log.groupby(["Dataset", "Algoritmo"])["Aciertos (%)"].idxmax()
    df_log.loc[idx_max, "Destacado"] = True

    header = [html.Thead(html.Tr([html.Th(col) for col in df_log.columns if col != "Destacado"]))]
    rows = []
    for _, row in df_log.iterrows():
        estilo = {"backgroundColor": "#d4edda"} if row["Destacado"] else {}
        cells = [html.Td(str(row[col])) for col in df_log.columns if col != "Destacado"]
        rows.append(html.Tr(cells, style=estilo))

    tabla = dbc.Table(
        header + [html.Tbody(rows)],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
        style={
            "marginTop": "20px",
            "fontFamily": "monospace",
            "fontSize": "16px",
            "letterSpacing": "0.05em",
            "width": "100%",
            "tableLayout": "fixed",
        },
    )
    return tabla

@app.callback(
    Output("page-content", "children"),
    Input("_pages_location", "pathname")
)
def mostrar_pagina(pathname):
    if pathname == "/algorithm_use":
        return dmc.Paper(
            children=[
                algorithm_representation
            ],
            p="md",
            shadow="sm",
            radius="md",
        )

    if pathname == "/general_table":
        tabla=table_page(CSV_LOG_PATH)
        return dmc.Paper([tabla], p="md", shadow="sm", radius="md")
    if  pathname == "/comparison_datos_formas":
        figures= comparison_data(datasets["Formas"],"Formas",CSV_LOG_PATH)
        return figures_comparison(figures)
    if pathname == "/comparison_compound":
        figures = comparison_data(datasets["Compound"], "Compound", CSV_LOG_PATH)
        return figures_comparison(figures)
    if pathname == "/comparison_pathbased":
        figures = comparison_data(datasets["Pathbased"], "Pathbased", CSV_LOG_PATH)
        return figures_comparison(figures)

    return dmc.Text("Selecciona una opción del menú.")

"""
app.layout = html.Div([
    html.H3(html.H2("Visualizador de Datasets"),


                        # Selector de dataset

                        "Selecciona los datos y el metodo de clusterizacion"),
                        dcc.Dropdown(
                                id='data_dropdown',
                                options=[{'label': k, 'value': k} for k in datasets.keys()],
                                value=list(datasets.keys())[0]
                            ),
                        html.Div(id='dataset_selected'),
                        html.Label("Selecciona el algoritmo de clustering"),
                        dcc.Dropdown(
                            id='algorithm_dropdown',
                            options=[{'label': k, 'value': k} for k in clust_algorithms],
                            value='none'  # valor inicial, será actualizado dinámicamente
                        ),
                        dcc.Store(id='store-dbscan-parametros'),
                        dcc.Store(id='clustered_data'),
                        dbc.Button("Abrir/cerrar parámetros algoritmos", id="btn-popover", n_clicks=0),
                        dbc.Popover(
                                [
                                    dbc.PopoverHeader("Parámetros algoritmos"),
                                    dbc.PopoverBody([
                                        dbc.PopoverBody(id="popover-body"),  # ← Este es dinámico

                                    ])
                                ],
                                id="popover-dbscan",
                                target="btn-popover",  # Ancla el popover al botón
                               # trigger="click",       # Se abre al hacer clic
                                placement="bottom",    # Aparece debajo del botón
                                is_open=False,          # Estado inicial cerrado
                                style={
                                        "backgroundColor": "white",
                                        "padding": "15px",
                                        "borderRadius": "10px",
                                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)",
                                        "zIndex": 2000  # Asegura que se superponga a lo de detrás
                                    }
                            ),



                       html.Hr(),
                       html.H3("Datos de los puntos seleccionados"),
                       #dcc.Graph(id="scatter-plot"),
                        html.Div([
                            html.Div([
                                dcc.Graph(id='scatter-plot-solution')
                            ], style={"width": "50%", "display": "inline-block"}),

                            html.Div([
                                dcc.Graph(id='scatter-plot-algorithm')
                            ], style={"width": "50%", "display": "inline-block"})
                        ], style={"display": "flex"}),
                       dcc.Graph(id="matriz"),
                       html.H3("Asignar nuevo grupo a los puntos seleccionados"), #arreglar
                       dcc.Input(id="nuevo-grupo", type="text", placeholder="Ingresa el nuevo grupo"),
                       html.Button("Asignar grupo", id="asignar-grupo-btn"),
                       html.Div(id="confirmacion-asignacion"),
                       html.Button("Guardar resultado", id="btn-guardar", n_clicks=0, className="btn btn-success"),
                       html.Div(id="tabla-resultados"),
                       html.Button("Calcular porcentaje", id="calcular_porcentaje"),
                       html.Div(id="porcentaje")
                       ])
"""

#Callback para abrir/cerrar popover
@app.callback(
    Output("popover-dbscan", "is_open"),
    Input("btn-popover", "n_clicks"),
    State("popover-dbscan", "is_open"),
    prevent_initial_call=True
)
def toggle_popover(n_clicks, is_open):
    return not is_open

@app.callback(
    Output("popover-body", "children"),
    Input("algorithm_dropdown", "value")
)
def actualizar_contenido_popover(algoritmo):

    if algoritmo not in parametros_por_algoritmo:
        return html.Div("Este algoritmo no tiene configuración")

    inputs = []
    for param in parametros_por_algoritmo[algoritmo]:
        row = dbc.Row([
            dbc.Col(html.Label(param + ":"), width="auto"),
            dbc.Col(
                dcc.Input(
                    id={"type": "param-clustering", "param": param},
                    type="number",
                    placeholder=param,
                    style={"width": "100%"}
                ),
                width=10
            )
        ], className="mb-2")
        inputs.append(row)

    inputs.append(
        dbc.Button("Guardar", id="guardar-config", size="sm", color="primary")
    )

    return html.Div(inputs)

# Callback para mostrar el número introducido
@app.callback(
    Output('store-dbscan-parametros', 'data'),
    Input("guardar-config", "n_clicks"),
    State({"type": "param-clustering", "param": ALL}, "id"),
    State({"type": "param-clustering", "param": ALL}, "value"),

    prevent_initial_call=True
)
def display_popover(n_clicks, ids,valores):
    parametros = {}

    for id_obj, valor in zip(ids, valores):
        if valor is not None:
            parametros[id_obj["param"]] = valor

    return parametros

@app.callback(
    Output('scatter-plot-algorithm','figure'),
    Output('scatter-plot-solution','figure'),
    Output('clustered_data','data'),
    Input('data_dropdown', 'value'),
    Input('algorithm_dropdown', 'value'),
    Input('store-dbscan-parametros', 'data')  # ← aquí lo traes

)
def print_dots(dataset_value,algorithm,dbscan_params):
    df=datasets[dataset_value]

    eps = dbscan_params.get('eps', 1.1) if dbscan_params else 1.1
    n_min = dbscan_params.get('n_min', 5) if dbscan_params else 5
    d_percent=dbscan_params.get('d_percent',1)if dbscan_params else 1
    n_clusters=dbscan_params.get('n_clusters',1)if dbscan_params else 1
    influencia = dbscan_params.get('sigma', 1) if dbscan_params else 1

    coords=df[['x','y']]

    if "grupo_manual" not in df.columns:
        df["grupo_manual"] = df["group"]
    figure_solution = px.scatter(df, x="x", y="y", color="group",
                                hover_data=["group", "grupo_manual"])
    if algorithm == 'none':

        figure_algorithm = px.scatter(df, x="x", y="y", color="group",
                            hover_data=["group","grupo_manual"])
    elif algorithm == 'DBscan':
        dbscan = DBSCAN(eps=eps, min_samples=n_min)
        df["grupo_clust"] = dbscan.fit_predict(coords)
        #figure_algorithm = px.scatter(df, x="x", y="y", color="grupo_clust",hover_data=["group", "grupo_manual", "grupo_clust"])
    elif algorithm =='HDBscan':
        df = df.sort_values(by=["x", "y"]).reset_index(drop=True)
        hdbscan_a = hdbscan.HDBSCAN(min_cluster_size=n_min)
        df["grupo_clust"] = hdbscan_a.fit_predict(df)
        #figure_algorithm = px.scatter(df, x="x", y="y", color="grupo_clust",hover_data=["group", "grupo_manual", "grupo_clust"])
    elif algorithm =='Density peak':
        df = df.sort_values(by=["x", "y"]).reset_index(drop=True)

        X = df[['x', 'y']].values
        distances = pairwise_distances(X)
        dc = np.percentile(distances, d_percent)
        rho = np.sum(distances < dc, axis=1) - 1
        delta = np.zeros_like(rho, dtype=float)
        nearest_higher = np.zeros_like(rho, dtype=int)

        for i in range(len(X)):
            higher = np.where(rho > rho[i])[0]
            if len(higher) > 0:
                j = higher[np.argmin(distances[i, higher])]
                delta[i] = distances[i, j]
                nearest_higher[i] = j
            else:
                delta[i] = np.max(distances[i])
                nearest_higher[i] = i
            # Paso 3: Calcular gamma
        gamma = rho * delta
        centers = np.argsort(gamma)[-n_clusters:]

        # Paso 4: Asignar clústeres
        labels = -np.ones(len(X), dtype=int)
        for idx, c in enumerate(centers):
            labels[c] = idx

        order = np.argsort(-rho)
        for i in order:
            if labels[i] == -1:
                labels[i] = labels[nearest_higher[i]]
        df["grupo_clust"]=labels
        #figure_algorithm = px.scatter(df, x="x", y="y", color="grupo_clust",hover_data=["group", "grupo_manual", "grupo_clust"])

    elif algorithm == 'Quickshift':
        df = df.sort_values(by=["x", "y"]).reset_index(drop=True)

        X = df[['x', 'y']].values
        n = len(X)

        # Estimación de densidad por kernel Gaussiano
        nbrs = NearestNeighbors(radius=3 * influencia).fit(X)
        radius_neighbors = nbrs.radius_neighbors(X, return_distance=True)

        densities = np.zeros(n)
        for i in range(n):
            distances_i = radius_neighbors[0][i]
            densities[i] = np.sum(np.exp(- (distances_i ** 2) / (2 * influencia ** 2)))

        # Conexión a vecino más cercano con mayor densidad
        parents = np.full(n, -1)
        for i in range(n):
            neighbors = radius_neighbors[1][i]
            valid = [j for j in neighbors if densities[j] > densities[i]]
            if valid:
                j_min = valid[np.argmin(np.linalg.norm(X[valid] - X[i], axis=1))]
                parents[i] = j_min
            else:
                parents[i] = i  # raíz

        # Asignar etiquetas de clúster a cada raíz distinta
        cluster_map = {}
        labels = np.full(n, -1)
        cluster_id = 0

        for i in range(n):
            # Seguir el árbol hasta la raíz
            path = []
            node = i
            while parents[node] != node:
                path.append(node)
                node = parents[node]
            root = node

            # Crear nuevo clúster si no existe
            if root not in cluster_map:
                cluster_members = [j for j in range(n) if parents[j] == root or j == root]
                if len(cluster_members) >= n_min:
                    cluster_map[root] = cluster_id
                    cluster_id += 1
                else:
                    cluster_map[root] = -1  # ruido

            labels[i] = cluster_map[root]
        df["grupo_clust"] = labels
    if "grupo_clust" not in df:
        df['grupo_clust']=df['group']
   # df['grupo_clust']=df['grupo_clust']+2

    if "grupo_clust" not in df:
        df['grupo_clust']=0
    if 'group' not in df.columns and 'grupo_manual' in df.columns:
        df['group'] = df['grupo_manual']
    # 1. Crear matriz de confusión (clustering vs real)
    real_labels = df['group'].unique()
    clust_labels = df['grupo_clust'].unique()

    n_real = len(real_labels)
    n_clust = len(clust_labels)
    if n_real < n_clust:
        n_to_add = n_clust - n_real

        max_real = df['group'].max()
        for i in range(n_to_add):
            nuevo_grupo = max_real + i + 1
            fila_ficticia = df.iloc[0].copy()
            fila_ficticia['group'] = nuevo_grupo
            df = pd.concat([df, pd.DataFrame([fila_ficticia])], ignore_index=True)

    matrix = pd.crosstab(df['grupo_clust'], df['group'])
    # 2. Convertir a numpy para usar el algoritmo húngaro
    matriz_numpy = matrix.to_numpy()

    # 3. Aplicar algoritmo (¡invertimos el signo para maximizar aciertos!)
    fila_ind, col_ind = linear_sum_assignment(-matriz_numpy)

    # 4. Crear el mapeo: grupo del clustering → grupo real
    grupos_cluster = matrix.index.to_numpy()
    grupos_reales = matrix.columns.to_numpy()

    mapeo = {grupos_cluster[fila]: grupos_reales[col] for fila, col in zip(fila_ind, col_ind)}

    # 5. Reasignar los grupos del clustering
    df['grupo_clust_reasignado'] = df['grupo_clust'].map(mapeo)
    grupos_originales = sorted(df['grupo_clust_reasignado'].dropna().unique())

    # Crear un nuevo mapeo secuencial: {original → nuevo}
    orden_cluster = [k for k, _ in sorted(mapeo.items(), key=lambda item: item[1])]
    mapeo_final = {cluster_id: i + 1 for i, cluster_id in enumerate(orden_cluster)}

    # Aplicar el mapeo final directamente al grupo_clust
    df['grupo_clust_reasignado'] = df['grupo_clust'].map(mapeo_final)
    df['correct_dot']=df['group']==df['grupo_clust_reasignado']


    figure_algorithm = px.scatter(df, x="x", y="y", color="grupo_clust",
                                  hover_data=["group", "grupo_clust_reasignado", "grupo_clust"])
    return figure_algorithm,figure_solution,df.to_dict("records")

@app.callback(
    Output('matriz','figure'),
    Input('data_dropdown', 'value'),
    Input('clustered_data', 'data'),

    Input('algorithm_dropdown', 'value'),
    Input('store-dbscan-parametros', 'data')  # ← aquí lo traes

)
def print_matrix(data_drop_down,data_,algorithm_drop_down,params):

    df=pd.DataFrame(data_)
    if df is None:
        df=datasets[data_drop_down].copy()
        process=algorithm_drop_down
        param=params



    #df['grupo_clust_reasignado'] = df['grupo_clust_reasignado'].fillna(df['group'])

    grupos_reasignados= sorted(df['grupo_clust_reasignado'].unique())
    grupos_clust=sorted(df['grupo_clust'].unique())
    grupos_reales=sorted(df['group'].unique())

    for g in range(1, 8):  # de 1 a 7 inclusive
        count = (df['grupo_clust_reasignado'] == g).sum()


    n_grupos = df['group'].nunique()
    n_grupos_reasignados = df['grupo_clust_reasignado'].nunique()

    if n_grupos_reasignados > n_grupos:
        size = grupos_reasignados
    else:
        size = grupos_reales


    matriz_final=pd.crosstab(df['group'],df['grupo_clust_reasignado'])
    #matriz_final = matriz_final.reindex(columns=todos_los_grupos_clust, fill_value=0)
   # matriz_final = matriz_final.reindex(index=size, columns=size, fill_value=0)


    todos_los_grupos = sorted(set(grupos_reales).union(set(grupos_reasignados)))
    matriz_final = matriz_final.reindex(index=todos_los_grupos, columns=todos_los_grupos, fill_value=0)

    fig = pgo.Figure(data=pgo.Heatmap(
        z=matriz_final.values,
        x=matriz_final.columns,
        y=matriz_final.index,
        colorscale="Blues",
        text=matriz_final.astype(str).values,
        texttemplate="%{text}",
        colorbar=dict(title="Número de puntos")
    ))
    fig.update_layout(
        title="Matriz de comparación manual vs clustering",
        xaxis_title="Grupo algoritmo",
        yaxis_title="Grupo real",
        height=500
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(matriz_final.columns),
        ticktext=[str(int(v)) for v in matriz_final.columns]
    )

    fig.update_yaxes(
        tickmode='array',
        tickvals=list(matriz_final.index),
        ticktext=[str(int(v)) for v in matriz_final.index]
    )
    #fig.update_layout(xaxis_autorange='reversed')
    return fig

"""
@app.callback(
    Output("confirmacion-asignacion", "children"),
    Output("scatter-plot", "figure"),
    Input("asignar-grupo-btn", "n_clicks"),
    State("nuevo-grupo", "value"),
    State("scatter-plot", "selectedData"),
    prevent_initial_call=True
)
def asignar_grupo(n_clicks, nuevo_grupo, selectedData):
    print('assing group')
    print(selectedData)
    print('----------------------')
    if not nuevo_grupo:
        return "Por favor, ingresa un nuevo grupo.", dash.no_update

    if selectedData and "points" in selectedData:
        # Extraer los índices de los puntos seleccionados
        selected_indices = [point["pointIndex"] for point in selectedData["points"]]

        print('selected_indices')
        print(selected_indices)
        # Asignar el nuevo grupo a los puntos seleccionados
        datos.loc[selected_indices, "grupo_manual"] = nuevo_grupo
        print('datos')
        print(datos)

        # Actualizar el gráfico de dispersión con los nuevos grupos
        fig = px.scatter(datos, x="x", y="y", color="group", hover_data=["group","grupo_manual"])
        print('-------------')
        print(datos)
        return f"Grupo '{nuevo_grupo}' asignado correctamente.", fig
    else:
        return "No hay puntos seleccionados.", dash.no_update

    # Ejecutar la app

"""

@app.callback(
    #Output("selected-data", "figure"),

    Input("scatter-plot", "selectedData")
)
def update_graph(selectedData,dataset_value):
    df=datasets[dataset_value]

    # Inicializar un DataFrame con todos los grupos y conteos en 0
    todos_los_grupos = df["group"].astype(str).unique()
    group_initial = pd.DataFrame({"group": todos_los_grupos, "count": 0})

    # Obtener los puntos seleccionados (si los hay)
    selected_points = selectedData["points"] if selectedData and "points" in selectedData else []

    if selected_points:
        # Extraer los índices de los puntos seleccionados
        selected_indices = [point["pointIndex"] for point in selected_points]
        selected_df = df.iloc[selected_indices].copy()
        selected_df["group"] = selected_df["group"].astype(str)

        # Contar los puntos seleccionados por grupo
        group_counts = selected_df["group"].value_counts().reset_index()
        group_counts.columns = ["group", "count"]


        # Combinar con el DataFrame inicial para incluir todos los grupos
        group_merge = group_initial.merge(group_counts, on="group", how="left", suffixes=("_initial", "_selected"))
        group_merge["count"] = group_merge["count_selected"].fillna(0).astype(int)
        group_merge.drop(columns=["count_selected"], inplace=True)
    else:
        # Si no hay puntos seleccionados, usar el DataFrame inicial
        group_merge = group_initial

    # Convertir la columna "group" a numérica para ordenar
    group_merge["group"] = pd.to_numeric(group_merge["group"])

    # Ordenar el DataFrame por la columna "group"
    # group_merge = group_merge.sort_values(by="group")

    # Crear el gráfico de barras
    return px.bar(group_merge, x="group", y="count", text_auto=True, title="Número de puntos seleccionados")

@app.callback(
    Output("porcentaje", "children"),
    Input("calcular_porcentaje", "n_clicks"),
    prevent_initial_call=True
)
def percentage(click):
    aciertos = (datos["grupo_manual"] == datos["group"]).sum()

    porcent=aciertos/len(datos)*100

    return "El porcentaje de acierto es: ",porcent,"%"
default_params = {
        "DBscan": {"eps": 1.1, "n_min": 5},
        "HDBscan": {"n_min": 5},
        "Density peak": {"d_percent": 1, "n_clusters": 1},
        "Quickshift": {"sigma": 1.0, "n_min": 5}
    }
@app.callback(
    Output("store_guardado", "data"),
    Input("btn-guardar", "n_clicks"),
    State("data_dropdown", "value"),
    State("algorithm_dropdown", "value"),
    State("store-dbscan-parametros", "data"),
    State("clustered_data", "data"),
    prevent_initial_call=True
)
def guardar_y_mostrar(n_clicks, dataset, algoritmo, parametros, datos_clusterizados):
    if datos_clusterizados is None:
        return "❌ No hay datos para guardar"

    if not parametros:
        parametros = default_params.get(algoritmo, {})

    df = pd.DataFrame(datos_clusterizados)

    # Calcular porcentaje de aciertos
    if "correct_dot" in df.columns:
        porcentaje = round(100 * df["correct_dot"].mean(), 2)
    else:
        porcentaje = "N/A"

    fila = {
        "Dataset": dataset,
        "Algoritmo": algoritmo,
        "Parametros": parametros,
        "Aciertos (%)": porcentaje
    }

    # Añadir al CSV
    #df_log = pd.read_csv(CSV_LOG_PATH)
    df_log = pd.read_csv(CSV_LOG_PATH, encoding="latin1")
    df_log = pd.concat([df_log, pd.DataFrame([fila])], ignore_index=True)
    df_log.to_csv(CSV_LOG_PATH, index=False)
    # Ordenar por Dataset y Algoritmo
    df_log["Aciertos (%)"] = pd.to_numeric(df_log["Aciertos (%)"], errors="coerce")
    df_log.sort_values(by=["Dataset", "Algoritmo", "Aciertos (%)"], ascending=[True, True, False], inplace=True)

    # Marcar las filas con mayor acierto por grupo
    df_log["Destacado"] = False
    idx_max = df_log.groupby(["Dataset", "Algoritmo"])["Aciertos (%)"].idxmax()
    df_log.loc[idx_max, "Destacado"] = True

    # Crear tabla HTML resaltando fila con mayor acierto
    header = [
        html.Thead(html.Tr([html.Th(col) for col in df_log.columns if col != "Destacado"]))
    ]

    rows = []
    for _, row in df_log.iterrows():
        estilo = {"backgroundColor": "#d4edda"} if row["Destacado"] else {}
        cells = [html.Td(str(row[col])) for col in df_log.columns if col != "Destacado"]
        rows.append(html.Tr(cells, style=estilo))

    tabla = dbc.Table(
        header + [html.Tbody(rows)],
        bordered=True,
        striped=True,
        hover=True,
        style={
            "marginTop": "20px",
            "fontFamily": "monospace",
            "fontSize": "16px",
            "letterSpacing": "0.05em",
            "width": "100%",
            "tableLayout": "fixed",
        },
        responsive=True
    )

    return tabla
if __name__ == "__main__":
    app.run(debug=True)

