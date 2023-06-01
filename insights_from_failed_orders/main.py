import folium
import geopandas
import h3
import pandas as pd
from folium.plugins import MarkerCluster
from shapely.geometry import Polygon


def read_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def make_failed_orders_map(df: pd.DataFrame) -> folium.Map:
    lat = df["origin_latitude"]
    lon = df["origin_longitude"]

    m = folium.Map(
        location=[sum(lat) / len(lat), sum(lon) / len(lon)],
        zoom_start=11,
        tiles="cartodbpositron",
    )
    marker_cluster = MarkerCluster().add_to(m)
    for lat, lon in zip(lat, lon):
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
        ).add_to(marker_cluster)
    return m


def count_orders_in_h3_cell(df: pd.DataFrame) -> pd.DataFrame:
    df["h3_cell"] = df.apply(
        lambda row: h3.geo_to_h3(row["origin_latitude"], row["origin_longitude"], 8),
        axis=1,
    )
    df = df.groupby(["h3_cell"], as_index=False)["order_gk"].agg(cnt="count")
    return df


def add_geometry(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    df["geometry"] = df.apply(
        lambda row: Polygon(h3.h3_to_geo_boundary(row["h3_cell"], True)), axis=1
    )
    gdf = geopandas.GeoDataFrame(df, crs="EPSG:4326")
    return gdf


def make_choropleth_maps(gdf: geopandas.GeoDataFrame) -> folium.Map:
    m = gdf.explore(
        column="cnt",
        cmap="YlOrRd",
        tiles="OpenStreetMap",
        legend_kwds={"caption": "Failed cab orders"},
        style_kwds={"fillOpacity": 0.8},
    )
    return m


def main():
    df = read_dataset("datasets/data_orders.csv")
    make_failed_orders_map(df).save('maps/failed_orders_map.html')
    orders_in_h3_cells = count_orders_in_h3_cell(df)
    gdf = add_geometry(orders_in_h3_cells)
    make_choropleth_maps(gdf).save('maps/choropleth.html')


if __name__ == "__main__":
    main()
