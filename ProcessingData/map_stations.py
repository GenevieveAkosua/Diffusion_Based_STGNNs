import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings

# Suppress standard geopandas centroid warnings for geographic coordinate systems
warnings.filterwarnings('ignore', 'GeoSeries.centroid')

def generate_static_map(file_path="station_metadata.csv", output_file="saws_stations_map.png"):
    print(f"Loading {file_path}...")
    
    try:
        # Load the station metadata
        df = pd.read_csv(file_path)

        # Convert coordinates into Shapely Point objects
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        geo_df = gpd.GeoDataFrame(df, geometry=geometry)

        # Load the Natural Earth world map directly (for neighboring countries)
        print("Downloading base map data...")
        world_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(world_url)

        # Load the Natural Earth states and provinces map
        print("Downloading provincial boundary data (this may take a few seconds)...")
        prov_url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
        provinces = gpd.read_file(prov_url)

        # Filter for South Africa and neighboring countries
        southern_africa = world[world['ADMIN'].isin([
            'South Africa', 'Lesotho', 'Swaziland', 'eSwatini', 'Namibia', 'Botswana', 'Zimbabwe', 'Mozambique'
        ])]
        
        # Filter the provinces specifically for South Africa
        sa_provinces = provinces[provinces['admin'] == 'South Africa']

        # Initialize a high-resolution plot (300 DPI)
        fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
        
        # Plot the base map (Neighboring countries)
        southern_africa.plot(ax=ax, color='#e8f4f8', edgecolor='#bdc3c7', linewidth=0.8)
        
        # Plot the South African provinces on top with slightly darker borders
        sa_provinces.plot(ax=ax, color='#e8f4f8', edgecolor='#7f8c8d', linewidth=1.2)

        # Add text labels for each province
        for idx, row in sa_provinces.iterrows():
            ax.annotate(
                text=row['name'], 
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                xytext=(0, 0), 
                textcoords="offset points",
                fontsize=8, 
                color='#7f8c8d', 
                fontweight='bold', 
                alpha=0.9,
                ha='center', 
                va='center'
            )

        # Plot the station nodes on top with a contrasting coral/red color
        geo_df.plot(ax=ax, color='#ff6b6b', markersize=60, marker='o', 
                    edgecolor='#c0392b', linewidth=1.2, zorder=5)

        # Set the bounding box exactly around South Africa
        ax.set_xlim([15, 34])
        ax.set_ylim([-36, -21])

        # Typography and aesthetics
        ax.set_title("SAWS Weather Station Nodes by Province", fontsize=16, fontweight='bold', pad=15, color='#2c3e50')
        ax.set_xlabel("Longitude", fontsize=12, color='#34495e')
        ax.set_ylabel("Latitude", fontsize=12, color='#34495e')
        
        # Set a soft off-white background color
        ax.set_facecolor('#fdfbf7')
        fig.patch.set_facecolor('#fdfbf7')

        # Add a subtle grid
        ax.grid(True, linestyle=':', color='#bdc3c7', alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color('#bdc3c7')

        # Save the map as a PNG
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Success! Provincial map saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Run the flow script first to generate it.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_static_map()
