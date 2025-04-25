import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, MATCH,ALL
import dash_bootstrap_components as dbc

"""
Añadir botones para ir eligiendo que algoritmo se representa y diferentes datos.
"""

parametros_por_algoritmo = {
    "DBscan": ["eps", "n_min"],
    "HDBscan": ["n_min"],
    "otro_algo": ["rango", "umbral"]
}
datos_formas = pd.read_csv("datos.txt")
datos = datos_formas.copy()
data_compound=pd.read_csv("compound.txt")
pathbased=pd.read_csv("pathbased_1")
clust_algorithms=['none','DBscan','HDBscan']
datasets={
    'Pathbased':pathbased,
    'Formas': datos,
    'Compound':data_compound

}
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

app.layout = html.Div([html.H3(html.H2("Visualizador de Datasets"),


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
                       html.H3("Asignar nuevo grupo a los puntos seleccionados"),
                       dcc.Input(id="nuevo-grupo", type="text", placeholder="Ingresa el nuevo grupo"),
                       html.Button("Asignar grupo", id="asignar-grupo-btn"),
                       html.Div(id="confirmacion-asignacion"),
                       html.Button("Calcular porcentaje", id="calcular_porcentaje"),
                       html.Div(id="porcentaje")
                       ])

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
        inputs.append(
            dcc.Input(
                id={"type": "param-clustering", "param": param},
                type="number",
                placeholder=param,
                style={"width": "100%", "marginBottom": "10px"}
            )
        )

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
    Input('data_dropdown', 'value'),
    Input('algorithm_dropdown', 'value'),
    Input('store-dbscan-parametros', 'data')  # ← aquí lo traes

)
def print_dots(dataset_value,algorithm,dbscan_params):
    df=datasets[dataset_value]
    print(dbscan_params)
    if dbscan_params:
        print("Recibe parametros")
    eps = dbscan_params.get('eps', 1.1) if dbscan_params else 1.1
    n_min = dbscan_params.get('n_min', 5) if dbscan_params else 5
    print("eps: ",eps)
    print("n_min: ",n_min)
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
        figure_algorithm = px.scatter(df, x="x", y="y", color="grupo_clust",
                            hover_data=["group", "grupo_manual", "grupo_clust"])
    elif algorithm ==('HDBscan'):
        hdbscan_a = hdbscan.HDBSCAN(min_cluster_size=n_min)
        df["grupo_clust"] = hdbscan_a.fit_predict(df)
        figure_algorithm = px.scatter(df, x="x", y="y", color="grupo_clust",
                            hover_data=["group", "grupo_manual", "grupo_clust"])
    return figure_algorithm,figure_solution

@app.callback(
    #Output("selected-data", "figure"),

    Input("scatter-plot", "selectedData")
)
def update_graph(selectedData,dataset_value):
    df=datasets[dataset_value]

    print('update graph')
    print(selectedData)
    print('datos')
    print(df)
    # Inicializar un DataFrame con todos los grupos y conteos en 0
    todos_los_grupos = df["group"].astype(str).unique()
    group_initial = pd.DataFrame({"group": todos_los_grupos, "count": 0})

    # Obtener los puntos seleccionados (si los hay)
    selected_points = selectedData["points"] if selectedData and "points" in selectedData else []
    print(f"Número de puntos seleccionados: {len(selected_points)}")

    print('selected_points: ')
    print(selected_points)
    if selected_points:
        # Extraer los índices de los puntos seleccionados
        selected_indices = [point["pointIndex"] for point in selected_points]
        selected_df = df.iloc[selected_indices].copy()
        selected_df["group"] = selected_df["group"].astype(str)

        # Contar los puntos seleccionados por grupo
        group_counts = selected_df["group"].value_counts().reset_index()
        group_counts.columns = ["group", "count"]
        print("Conteo de grupos seleccionados:")
        print(group_counts)

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

    print("DataFrame final:")
    print(group_merge)
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
if __name__ == "__main__":
    app.run(debug=True)

