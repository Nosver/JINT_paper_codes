import matplotlib.pyplot as plt
import pandas as pd
from ortools.constraint_solver import pywrapcp
import numpy as np

from ortools.constraint_solver import routing_enums_pb2  # Bunu ekle
from ortools.constraint_solver import pywrapcp
import os

def tsp_solver(locations):
   
    locations = np.array(locations)

    def compute_euclidean_distance_matrix(locations):
        distance_matrix = np.zeros((len(locations), len(locations)))
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(locations[i] - locations[j])
        return distance_matrix

    distance_matrix = compute_euclidean_distance_matrix(locations)

    def create_data_model():
        data = {}
        data['distance_matrix'] = distance_matrix.tolist()
        data['num_vehicles'] = 1
        data['depot'] = 0
        return data

    data = create_data_model()

    # Sadece başlangıç noktasını veriyoruz
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Bitiş noktasına geri dönülmesin:
    # Araç, herhangi bir noktada bitebilir (Open TSP)
    for node in range(1, len(locations)):
        routing.AddDisjunction([manager.NodeToIndex(node)], 100000)

    routing.SetFixedCostOfAllVehicles(0)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route_indices = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route_indices.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        route_coordinates = [tuple(locations[i]) for i in route_indices]
        return route_coordinates, solution.ObjectiveValue()
    else:
        return None, None


def write_route_to_txt(route, filename="outputs/tsp_route.txt"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        for x, y in route:
            file.write(f"{x:.4f} {y:.4f} 0\n")
    print("Successfully printed to txt")

def plot_route(route):
    """
    Koordinatları sırayla oklarla birleştirip matplotlib ile çiz.
    """
    route = np.array(route)
    
    plt.figure(figsize=(10, 8))
    plt.plot(route[:, 0], route[:, 1], 'o-', color='blue', label="Rota")
    
    # Her noktayı numaralandır
    for i, (x, y) in enumerate(route):
        plt.text(x, y, f'{i}', fontsize=9, ha='right', va='bottom', color='darkred')
    
    # Oklarla gösterim
    for i in range(len(route) - 1):
        plt.arrow(route[i, 0], route[i, 1],
                  route[i+1, 0] - route[i, 0],
                  route[i+1, 1] - route[i, 1],
                  shape='full', lw=0, length_includes_head=True,
                  head_width=200, color='green', alpha=0.7)

    plt.title("TSP Rota Çizimi")
    plt.xlabel("X koordinatı")
    plt.ylabel("Y koordinatı")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# Verileri oluştur
data = {
    "x": [13200, 12719.14286, 12238.28571, 11757.42857, 11276.57143, 13135.02857,
          14993.48571, 16851.94286, 18710.4, 11299.2, 11404.8, 13886.4, 12196.8,
          10771.2, 7920, 15787.2, 14361.6, 9292.8, 22968, 23073.6, 17265.6, 26188.8,
          22334.4, 15892.8, 25766.4, 19694.4, 8606.4, 9873.6, 11457.6, 6916.8, 9662.4,
          10507.2, 12249.6, 14308.8],
    "y": [13200, 12983.14286, 12766.28571, 12549.42857, 12332.57143, 13918.62857,
          15504.68571, 17090.74286, 18676.8, 9820.8, 7444.8, 13992, 6969.6, 8764.8,
          15364.8, 5755.2, 11668.8, 8289.6, 15364.8, 17529.6, 18691.2, 20803.2,
          11035.2, 11985.6, 13780.8, 20064, 21648, 16526.4, 24763.2, 18216, 16209.6,
          24499.2, 21489.6, 17424]
}

df = pd.DataFrame(data)


CS = df.iloc[:9]  # ilk 9 nokta cs ler
WP = df.iloc[9:]

# Noktaları çiz
plt.figure(figsize=(10, 8))
plt.scatter(CS["x"], CS["y"], color='red', label='cs', s=50)
plt.scatter(WP["x"], WP["y"], color='blue', label='wp', s=50)
plt.title("Lokasyonlar (Son 8 Nokta Kırmızı ile Gösterildi)")
plt.xlabel("X Koordinatı")
plt.ylabel("Y Koordinatı")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()

route, total_distance = tsp_solver(WP)

plot_route(route)

print("En kısa rota (koordinatlar):")
for coord in route:
    print(coord)
print("Toplam mesafe:", total_distance)

write_route_to_txt(route)


coordinates = CS[['x', 'y']].values

# Mesafe hesaplama fonksiyonu
def compute_euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Fully connected graph'ı yazmak için fonksiyon
def write_fully_connected_graph_to_txt(coordinates, filename="fully_connected_graph.txt"):
    with open(filename, 'w') as f:
        for i, (x, y) in enumerate(coordinates):
            # Bağlantılı olduğu noktalar (herhangi bir noktaya bağlı)
            connected_nodes = [j for j in range(len(coordinates)) if j != i]  # Kendisiyle bağlanmaz
            
            # Bağlantılı noktaları yazalım
            line = f"{i} {x:.3f} {y:.3f} 0 "  # 0: Z koordinatı sabit
            line += " ".join(map(str, connected_nodes)) + " L\n"
            f.write(line)

# Dosyaya yaz
write_fully_connected_graph_to_txt(coordinates)
