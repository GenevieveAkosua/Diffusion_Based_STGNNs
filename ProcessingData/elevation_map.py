import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings

# Suppress standard geopandas centroid warnings
warnings.filterwarnings('ignore', 'GeoSeries.centroid')

def generate_elevation_map(file_path="station_metadata.csv", output_file="saws_elevation_map.png"):
    print(f"Loading {file_path}...")
    
    try:
        # Load the station metadata
        df = pd.read_csv(file_path)

        # Load the Natural Earth world map directly
        print("Downloading base map data...")
        world_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(world_url)

        # Load the Natural Earth states and provinces map
        print("Downloading provincial boundary data...")
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
        
        # Plot the base map and provinces
        southern_africa.plot(ax=ax, color='#e8f4f8', edgecolor='#bdc3c7', linewidth=0.8)
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

        # Plot the station nodes, using the 'height' column for color
        # 'plasma' is a beautiful, colorblind-friendly colormap (dark purple to bright yellow)
        scatter = ax.scatter(
            df['longitude'], 
            df['latitude'], 
            c=df['height'], 
            cmap='plasma', 
            s=80, 
            edgecolor='#2c3e50', 
            linewidth=1, 
            zorder=5,
            alpha=0.9
        )

        # Add a colorbar legend to the side of the map
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Elevation / Height (meters)', fontsize=12, fontweight='bold', color='#2c3e50')
        cbar.ax.tick_params(labelsize=10, colors='#34495e')

        # Set the bounding box exactly around South Africa
        ax.set_xlim([15, 34])
        ax.set_ylim([-36, -21])

        # Typography and aesthetics
        ax.set_title("SAWS Weather Station Elevations", fontsize=16, fontweight='bold', pad=15, color='#2c3e50')
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
        print(f"Success! Elevation map saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_elevation_map()
