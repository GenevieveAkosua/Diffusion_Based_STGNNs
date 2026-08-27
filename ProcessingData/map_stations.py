import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings

# Suppress standard geopandas centroid warnings
warnings.filterwarnings('ignore', 'GeoSeries.centroid')

def generate_bw_provincial_map(file_path="station_metadata.csv", output_file="saws_corrected_model_map.png"):
    print(f"Loading {file_path}...")
    
    try:
        # Load the station metadata
        df = pd.read_csv(file_path)

        # Convert coordinates into Shapely Point objects to ensure proper map alignment
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        geo_df = gpd.GeoDataFrame(df, geometry=geometry)

        # Load the Natural Earth world map
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
        
        # Plot the base map (light grey land, black borders)
        southern_africa.plot(ax=ax, color='#d3d3d3', edgecolor='black', linewidth=0.8)
        
        # Plot the South African provinces on top
        sa_provinces.plot(ax=ax, color='#d3d3d3', edgecolor='black', linewidth=0.8)

        # Add text labels for each province (using dark grey so it doesn't clash with the black text)
        for idx, row in sa_provinces.iterrows():
            ax.annotate(
                text=row['name'], 
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                xytext=(0, 0), 
                textcoords="offset points",
                fontsize=8, 
                color='#333333', 
                fontweight='bold', 
                alpha=0.8,
                ha='center', 
                va='center'
            )

        # Plot the station nodes as white circles with black borders
        geo_df.plot(ax=ax, color='white', markersize=300, marker='o', 
                    edgecolor='black', linewidth=1.2, zorder=5)

        # Add the station ID numbers inside the circles
        for idx, row in df.iterrows():
            ax.text(
                row['longitude'], 
                row['latitude'], 
                str(int(row['id'])), 
                fontsize=8, 
                color='black', 
                ha='center', 
                va='center', 
                zorder=6,
                fontweight='bold'
            )

        # Set the bounding box exactly around South Africa
        ax.set_xlim([15, 34])
        ax.set_ylim([-36, -21])

        # Clean aesthetics matching the model map
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Remove axes ticks for the clean look, or remove these two lines to keep longitude/latitude labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Ensure a clean black border around the whole map
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

        # Save the map as a PNG
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Success! Map saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_bw_provincial_map()
