import pandas as pd
import numpy as np

# Semilla para reproducibilidad
np.random.seed(42)

# Configuración del dataset
n_players = 200
names = [
    "Lionel Messi", "Cristiano Ronaldo", "Kylian Mbappé", "Erling Haaland", 
    "Kevin De Bruyne", "Mohamed Salah", "Robert Lewandowski", "Vinícius Jr", 
    "Jude Bellingham", "Harry Kane", "Bukayo Saka", "Antoine Griezmann", 
    "Luka Modrić", "Martin Ødegaard", "Rodri", "Bruno Fernandes", 
    "Son Heung-min", "Lautaro Martínez", "Phil Foden", "Pedri"
] + [f"Player_{i}" for i in range(21, n_players + 1)]

teams = ["Real Madrid", "FC Barcelona", "Manchester City", "Liverpool", "Bayern Munich", "PSG", "Inter Milan", "Arsenal", "Atletico Madrid", "Bayer Leverkusen"]
leagues = ["La Liga", "Premier League", "Bundesliga", "Serie A", "Ligue 1"]

# Generación de datos aleatorios
data = {
    'Nombre': names,
    'Edad': np.random.randint(18, 38, n_players),
    'Equipo': [np.random.choice(teams) for _ in range(n_players)],
    'Liga': [np.random.choice(leagues) for _ in range(n_players)],
    'Valor_Mercado': np.random.randint(5, 180, n_players),
    'Goles': np.random.randint(0, 35, n_players),
    'Asistencias': np.random.randint(0, 20, n_players),
    'Pases_%': np.random.uniform(70, 95, n_players).round(1),
    'Regates': np.random.randint(0, 150, n_players),
    'Recuperaciones': np.random.randint(0, 200, n_players),
    'Duelos_Aereos': np.random.randint(0, 100, n_players),
    'xG': np.random.uniform(0.5, 30.0, n_players).round(1)
}

df = pd.DataFrame(data)

# Guardar a CSV
df.to_csv("players_data.csv", index=False)
print("Dataset 'players_data.csv' generado con éxito en el directorio actual.")
