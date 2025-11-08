import math
import random
from itertools import combinations
from collections import defaultdict 
import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

class Util:
    @staticmethod
    def divide_into_quarters(grid):
        mid_row = len(grid) // 2
        mid_col = len(grid[0]) // 2
    
        quarters = { 
            "Q1": [row[:mid_col] for row in grid[:mid_row]],
            "Q2": [row[mid_col:] for row in grid[:mid_row]],
            "Q3": [row[:mid_col] for row in grid[mid_row:]], 
            "Q4":  [row[mid_col:] for row in grid[mid_row:]]
        }
    
        return quarters
  

    @staticmethod
    def get_moore_neighbourhood(grid, x, y,radius = 1):
        height = len(grid)  # Ottieni il numero di righe della griglia
        width = len(grid[0])  # Ottieni il numero di colonne della griglia
        neighbours = []  # Lista per conservare i vicini

        # Itera su tutte le posizioni di vicinato in un raggio 1 attorno alla cella (x, y)
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                if dx == 0 and dy == 0:
                    continue  # Salta la cella centrale (x, y)

                # Calcola la posizione del vicino
                nx, ny = x + dx, y + dy

                # Verifica che il vicino sia dentro i limiti della griglia
                if 0 <= nx < height and 0 <= ny < width:
                    neighbours.append(grid[nx][ny])  # Aggiungi il vicino alla lista

        return neighbours  # Restituisci la lista dei vicini




    @staticmethod
    def calculate_Full_Theil_Index(ethnic_distribution, total_agents, quarters):
        p_g = {}
        for group in ethnic_distribution: 
            p_g[ethnic_distribution[group]["color"]] =  ethnic_distribution[group]["percentage"] #percentuale totale di ogni gruppo
        
        T = 0.0 

        #per ogni quartiere
        for quarter in quarters.values():
            local_counts = {} #quante persone di ogni colore per ogni quartiere
            n_j = 0  #conta quante persone ho nel quartiere
            
            #per ogni riga del quartiere
            for row in quarter:
                #per ogni agente nella riga
                for agent in row:
                    #se diverso da cella vuota
                    if agent.color != agent.empty:
                        #aumento il numero di agenti
                        n_j += 1
                        #salvo il numero di colori per ogni quartiere
                        local_counts[agent.color] = local_counts.get(agent.color, 0) + 1

            if n_j == 0:
                continue 

                
            #calcolo la proporzione di ogni gruppo nel quartiere
            p_gj = {color: count / n_j for color, count in local_counts.items()}
            #calcolo il peso
            weight = n_j / total_agents

            inner_sum = 0.0
            for color, pgj in p_gj.items():
                pg = p_g.get(color)
                if pg is None or pg == 0:
                    continue  # salta se non definito o zero (evita divisione o log indefinito)

                ratio = pgj / pg
                inner_sum += pgj * math.log2(ratio)


            T += weight * inner_sum

        return T
    @staticmethod
    def calculate_Local_Theil_Index(ethnic_distribution, total_agents, quarters):
        p_g = {}
        for group in ethnic_distribution: 
            p_g[ethnic_distribution[group]["color"]] =  ethnic_distribution[group]["percentage"] #percentuale totale di ogni gruppo
        
        T = {}

        #per ogni quartiere
        for (name, quarter) in quarters.items():
            local_counts = {} #quante persone di ogni colore per ogni quartiere
            n_j = 0  #conta quante persone ho nel quartiere
            
            #per ogni riga del quartiere
            for row in quarter:
                #per ogni agente nella riga
                for agent in row:
                    #se diverso da cella vuota
                    if agent.color != agent.empty:
                        #aumento il numero di agenti
                        n_j += 1
                        #salvo il numero di colori per ogni quartiere
                        local_counts[agent.color] = local_counts.get(agent.color, 0) + 1

            if n_j == 0:
                continue 

                
            #calcolo la proporzione di ogni gruppo nel quartiere
            p_gj = {color: count / n_j for color, count in local_counts.items()}
            #calcolo il peso

            inner_sum = 0.0
            for color, pgj in p_gj.items():
                pg = p_g.get(color)
                ratio = pgj / pg
                inner_sum += pgj * math.log2(pgj / pg)
                
            T[name] = inner_sum
        

        return T


    def gini_movements(array):
        arr = np.array(array)
        if np.amin(arr) < 0:
            arr -= np.amin(arr)  # Rendi tutto positivo
        arr = np.array(arr,dtype=float)
        arr += 1e-10  # Evita divisione per zero
        arr = np.sort(arr)
        n = arr.size
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))

 
    def single_gini(grid, ethnic_distribution):
        # Estrai i colori da ethnic_distribution
        colors_in_ethnic_distribution = [info["color"] for info in ethnic_distribution.values()]
        # Inizializza un dizionario per raccogliere i movimenti separati per colore
        movements_by_color = {color: [] for color in colors_in_ethnic_distribution}

        # Raccogli i movimenti separati per ciascun colore etnico
        for row in grid:
            for a in row:
                # Controlla che l'agente non sia vuoto e che il suo colore sia in ethnic_distribution
                if a.color in movements_by_color:
                    movements_by_color[a.color].append(a.movements)

        gini_by_color = {}

        # Calcola l'indice di Gini per ogni colore etnico
        for color, movements in movements_by_color.items():
            if len(movements) > 0:
                gini_by_color[color] = Util.gini_movements(movements)
            else:
                gini_by_color[color] = None  # Se non ci sono movimenti per quel colore, metti None
        
        return gini_by_color


       
    @staticmethod
    def calculate_Nagents(quarters):
        
        count_agents_in_quarters = []
        for _, quarter in quarters.items():
            count_agent = 0
            for row in quarter:
                for agent in row:
                    if agent.color != agent.empty:
                        count_agent+=1
            count_agents_in_quarters.append(count_agent)
                    
        return np.array(count_agents_in_quarters)
            
    @staticmethod
    def moving_average(data, window_size):
        if window_size % 2 == 0:  # finestra pari
            pad_left = window_size // 2
            pad_right = window_size // 2 - 1
        else:
            pad_left = pad_right = window_size // 2

        padded = np.pad(data, (pad_left, pad_right), mode='edge')
        return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')



    @staticmethod
    def moving_median(data, window_size=5):
        return median_filter(data, size=(window_size), mode='nearest')

    @staticmethod
    def inverse_log_sample_array(ma_flux, num_points=20):
        """
        Campionamento inverso logaritmico: meno frequente all'inizio, più frequente alla fine.
        """
        n = len(ma_flux)
        # Indici logaritmici (più fitti all'inizio)
        log_indices = np.logspace(0, np.log10(n - 1), num=num_points, dtype=int)
        log_indices = np.unique(log_indices)  # Rimuove duplicati
        # Inverti: meno fitti all'inizio, più fitti alla fine
        reversed_indices = np.array([n - 1 - i for i in reversed(log_indices)])
        reversed_indices = np.clip(reversed_indices, 0, n - 1)
        sampled_values = [ma_flux[i] for i in reversed_indices]
        if(n-1 not in reversed_indices):
            sampled_values.append(ma_flux[n-1])

        return sampled_values
    
    
    @staticmethod
    def plot_simulation_data(full_theil, local_theil, gini_by_color, ethnic_distribution,
                         unhappy_counts, average_tolerances, movement_counts, flux):

        steps = list(range(len(unhappy_counts)))
        window_size = 10  # puoi regolare la finestra

        ma_flux = Util.moving_average(np.array(flux), window_size)
        valid_flux_steps = steps[len(steps) - len(ma_flux):]

        fig, ax = plt.figure(figsize=(16, 12))
        line , = ax.plot([], [], 'ko-')
        # 1. Infelicità nel tempo
        plt.subplot(3, 2, 1)
        plt.plot(steps, unhappy_counts, color='red')
        plt.xlabel("Step")
        plt.ylabel("Percetage")
        plt.title("Unhappiness over time")
        plt.grid(True)

        # 2. Tolleranza media nel tempo
        plt.subplot(3, 2, 2)
        plt.plot(steps, average_tolerances, color='blue')
        plt.xlabel("Step")
        plt.ylabel("Tolerance")
        plt.title("Average tolerance over time")
        plt.grid(True)

        # 3. Movimenti degli agenti
        plt.subplot(3, 2, 3)
        plt.plot(steps, movement_counts, color='green')
        plt.xlabel("Step")
        plt.ylabel("Average movements")
        plt.title("Average movement per agent")
        plt.grid(True)

        # 4. Indice di Theil
        plt.subplot(3, 2, 4)
        bar_width = 0.35
        plt.bar(['Globale'], [full_theil], color='purple', width=bar_width, label='Theil Globale')
        local_keys = list(local_theil.keys())
        local_vals = list(local_theil.values())
        plt.bar(local_keys, local_vals, color='orange', width=bar_width, label='Theil Locale')
        plt.ylabel("Valore indice")
        plt.title("Theil's index (global and 'by quarter')")
        plt.legend()
        plt.grid(axis='y')

        # 5. Gini per etnia
        plt.subplot(3, 2, 5)
        color_to_ethnic = {v["color"]: k for k, v in ethnic_distribution.items()}
        labels = [color_to_ethnic[c] for c in gini_by_color]
        gini_values = list(gini_by_color.values())
        plt.bar(labels, gini_values, color='skyblue')
        plt.xlabel("Etnia")
        plt.ylabel("Gini's index")
        plt.title("Gini's index by Ethnicity")
        plt.grid(axis='y')

        # 6. Flusso migratorio
        plt.subplot(3, 2, 6)
        plt.plot(steps, flux, label="Instant flux", color="orange", alpha=0.4)
        plt.plot(valid_flux_steps, ma_flux, label="Moving average", color="blue", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Flux")
        plt.title("Average migratory flux")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
        
        
    import matplotlib.pyplot as plt

    @staticmethod
    def plot_simulation_data(full_theil, local_theil, gini_by_color, ethnic_distribution):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # === 1. Indice di Theil ===
        ax1 = axs[0]
        bar_width = 0.35

        # Barre globale e locale
        bars1 = ax1.bar(['Globale'], [full_theil], color='purple', width=bar_width, label='Theil Globale')
        local_keys = list(local_theil.keys())
        local_vals = list(local_theil.values())
        bars2 = ax1.bar(local_keys, local_vals, color='orange', width=bar_width, label='Theil Locale')

        local_keys = [list(local_theil.keys())]
        local_vals = list(local_theil.values())
        
        # Etichette sui valori
        ax1.bar_label(bars1, fmt='%.2f', padding=3)
        ax1.bar_label(bars2, fmt='%.2f', padding=3)

        ax1.set_ylabel("Index value")
        ax1.set_title("Theil's index (global and 'by quarter')")
        ax1.legend()
        ax1.grid(axis='y')
    

        # === 2. Gini per etnia ===
        ax2 = axs[1]
        color_to_ethnic = {v["color"]: k for k, v in ethnic_distribution.items()}
        labels = [color_to_ethnic[c] for c in gini_by_color]
        gini_values = list(gini_by_color.values())
        
        bars3 = ax2.bar(labels, gini_values, color='skyblue')
        ax2.bar_label(bars3, fmt='%.2f', padding=3)

        ax2.bar(labels, gini_values, color='skyblue')
        ax2.set_xlabel("Etnia")
        ax2.set_ylabel("Gini's index")
        ax2.set_title("Gini's index by ethnicity")
        ax2.grid(axis='y')

        plt.tight_layout()
        plt.show()

