import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

"""
Añadir botones para ir eligiendo que algoritmo se representa y diferentes datos.
"""


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
app = dash.Dash(__name__)

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
                        dbc.Button("Configurar DBSCAN", id="btn-popover", n_clicks=0),
                        dbc.Popover(
                                [
                                    dbc.PopoverHeader("Parámetros DBSCAN"),
                                    dbc.PopoverBody([
                                        dcc.Input(id='input-eps', type='number', placeholder='Epsilon', style={'width': '100%', 'marginBottom': '10px'}),
                                        dcc.Input(id='input-min', type='number', placeholder='Min samples', style={'width': '100%'}),
                                        html.Br(),
                                        html.Br(),
                                        dbc.Button("Guardar", id="guardar-numero", size="sm", color="primary")
                                    ])
                                ],
                                id="popover-dbscan",
                                target="btn-popover",  # Ancla el popover al botón
                               # trigger="click",       # Se abre al hacer clic
                                placement="bottom",    # Aparece debajo del botón
                                is_open=False          # Estado inicial cerrado
                            ),



                       html.Hr(),
                       html.H3("Datos de los puntos seleccionados"),
                       dcc.Graph(id="scatter-plot"),
                       html.H3("Asignar nuevo grupo a los puntos seleccionados"),
                       dcc.Input(id="nuevo-grupo", type="text", placeholder="Ingresa el nuevo grupo"),
                       html.Button("Asignar grupo", id="asignar-grupo-btn"),
                       html.Div(id="confirmacion-asignacion"),
                       html.Button("Calcular porcentaje", id="calcular_porcentaje"),
                       html.Div(id="porcentaje")
                       ])

"""
dbc.Button("Seleccion parametros", id="abrir-modal", n_clicks=0),
                        dbc.Modal([
                                dbc.ModalHeader("Introduce el valor de epsilon y el minimo de puntos"),
                                dbc.ModalBody([
                                    dcc.Input(id='input-eps', type='number', placeholder='Escribe epsilon',
                                              style={"width": "100%"}),
                                    dcc.Input(id='input-min', type='number', placeholder='Escribe el numero min', style={"width": "100%"}),
                                ]),
                                dbc.ModalFooter([
                                    html.Button("Aceptar", id="guardar-numero", n_clicks=0, className="btn btn-primary")
                                ]),
                            ], id="modal", is_open=False),
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
# Callback para abrir/cerrar modal
"""
@app.callback(
    Output("modal", "is_open"),
    [
        Input("abrir-modal", "n_clicks"),
        Input("guardar-numero", "n_clicks"),
    ],
    State("modal", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(open_click, guardar_click,  is_open):
    triggered = dash.ctx.triggered_id

    if triggered in ["abrir-modal", "guardar-numero"]:
        return not is_open
    return is_open
"""
# Callback para mostrar el número introducido
@app.callback(
    Output('store-dbscan-parametros', 'data'),
    Input("guardar-numero", "n_clicks"),
    State("input-min", "value"),
    State("input-eps", "value"),

    prevent_initial_call=True
)
def guardar_numero(n_clicks, n_min,eps):
    print(eps)
    if eps is None or n_min is None:
        return dash.no_update  # Evita guardar si algún campo está vacío
    else:
        print("guardado")
    return {'eps': eps, 'n_min': n_min}

@app.callback(
    Output('scatter-plot','figure'),
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
    if algorithm == 'none':

        figure = px.scatter(df, x="x", y="y", color="group",
                            hover_data=["group","grupo_manual"])
    elif algorithm == 'DBscan':
        dbscan = DBSCAN(eps=eps, min_samples=n_min)
        df["grupo_clust"] = dbscan.fit_predict(coords)
        figure = px.scatter(df, x="x", y="y", color="grupo_clust",
                            hover_data=["group", "grupo_manual", "grupo_clust"])
    elif algorithm ==('HDBscan'):
        hdbscan_a = hdbscan.HDBSCAN(min_cluster_size=10)
        df["grupo_clust"] = hdbscan_a.fit_predict(df)
        figure = px.scatter(df, x="x", y="y", color="grupo_clust",
                            hover_data=["group", "grupo_manual", "grupo_clust"])
    return figure
"""
@app.callback(
    Output('grafico', 'figure'),
    Input('data_dropdown', 'value'),
    Input('algorithm_dropdown', 'value')
)
def actualizar_grafico(dataset_seleccionado, columna_y):
    df = datasets[dataset_seleccionado]
    print('1')
    fig = px.scatter(df, x='x', y=columna_y, title=f'{columna_y} vs x en {dataset_seleccionado}')
    return fig
    """
"""
@app.callback(
    Output('dataset_selected', 'children'),
    Output('scatter-plot','figure'),
    Input('data_dropdown', 'value'),
    Input('algorithm_dropdown', 'value')
)
def Data_selection(valor_seleccionado):
    datos = datasets[valor_seleccionado]
    return f"Seleccionaste: {valor_seleccionado}, con datos: {datos}"
    """

@app.callback(
    Output("selected-data", "figure"),

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

