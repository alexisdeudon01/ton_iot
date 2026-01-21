#!/usr/bin/env python3
"""
Interface graphique interactive (Tkinter) pour visualiser en temps réel
l'exécution des algorithmes de classification réseau.

Auteurs: Système Expert IA
Date: 2025-01-21
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
import io
from datetime import datetime

# Imports pour les modèles
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Imports optionnels pour CNN et TabNet
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    import dtreeviz
    DTreeViz_AVAILABLE = True
except ImportError:
    DTreeViz_AVAILABLE = False

# Configuration du logging
LOG_FILE = Path("visualisation_algo.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlgorithmVisualizer:
    """
    Interface graphique principale pour la visualisation des algorithmes.
    """
    
    def __init__(self, root: tk.Tk):
        """Initialise l'interface graphique."""
        self.root = root
        self.root.title("Visualisateur d'Algorithmes de Classification Réseau")
        self.root.geometry("1400x900")
        
        # Données
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Modèle actuel
        self.current_model: Optional[Any] = None
        self.model_name: str = ""
        
        # Threading
        self.training_thread: Optional[threading.Thread] = None
        self.is_training: bool = False
        self.should_stop: bool = False
        self.is_paused: bool = False
        
        # Timing
        self.start_time: Optional[float] = None
        self.timings: Dict[str, float] = {}
        
        # Métriques en cours d'entraînement
        self.training_history: Dict[str, list] = {
            'loss': [],
            'accuracy': [],
            'epoch': []
        }
        
        # Setup UI
        self._setup_ui()
        
        logger.info("Interface graphique initialisée")
    
    def _setup_ui(self):
        """Configure l'interface utilisateur."""
        # Panneau principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration de la grille
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # === Panneau de contrôle gauche ===
        control_frame = ttk.LabelFrame(main_frame, text="Contrôles", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Chargement des données
        ttk.Label(control_frame, text="1. Charger les données:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="Charger CSV", command=self._load_data).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Sélection du modèle
        ttk.Label(control_frame, text="2. Sélectionner le modèle:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.model_var = tk.StringVar(value="Decision Tree")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                   values=["Decision Tree", "Random Forest", "Logistic Regression", 
                                          "CNN", "TabNet"], state="readonly")
        model_combo.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
        # Boutons de contrôle
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.run_button = ttk.Button(button_frame, text="▶ Lancer", command=self._start_training)
        self.run_button.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Arrêter", command=self._stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.pause_button = ttk.Button(button_frame, text="⏸ Pause", command=self._pause_training, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=2, sticky=(tk.W, tk.E))
        
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        # Timer
        ttk.Label(control_frame, text="3. Timer:").grid(row=5, column=0, sticky=tk.W, pady=(20, 5))
        self.timer_label = ttk.Label(control_frame, text="00:00:00.000", font=("Arial", 16, "bold"))
        self.timer_label.grid(row=6, column=0, pady=(0, 10))
        
        # Détails du timer
        self.timer_details = ttk.Label(control_frame, text="", font=("Arial", 9))
        self.timer_details.grid(row=7, column=0, pady=(0, 10))
        
        # === Zone de visualisation centrale ===
        viz_frame = ttk.LabelFrame(main_frame, text="Visualisation du Modèle", padding="10")
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Figure matplotlib
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Toolbar matplotlib
        toolbar = NavigationToolbar2Tk(self.canvas, viz_frame)
        toolbar.update()
        
        # Axe par défaut
        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, "Charger des données et sélectionner un modèle\ pour commencer", 
                     ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
        
        # === Zone de logs ===
        log_frame = ttk.LabelFrame(main_frame, text="Logs d'Exécution", padding="10")
        log_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Redirection des logs vers l'interface
        self._setup_log_handler()
    
    def _setup_log_handler(self):
        """Configure un handler pour afficher les logs dans l'interface."""
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                logging.Handler.__init__(self)
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                def append():
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.see(tk.END)
                self.text_widget.after(0, append)
        
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                                     datefmt='%H:%M:%S'))
        logger.addHandler(text_handler)
    
    def _load_data(self):
        """Charge un fichier CSV via une boîte de dialogue."""
        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            logger.info(f"Chargement du fichier: {file_path}")
            self._log_message("🔄 Chargement des données...")
            
            # Chargement du CSV
            self.data = pd.read_csv(file_path)
            
            # Détection automatique de la colonne cible (dernière colonne ou 'label')
            if 'label' in self.data.columns:
                target_col = 'label'
            else:
                target_col = self.data.columns[-1]
            
            # Séparation features/target
            X = self.data.drop(columns=[target_col])
            y = self.data[target_col]
            
            # Encodage si nécessaire
            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Nettoyage des features (gestion des NaN)
            X = X.select_dtypes(include=[np.number])
            X = X.fillna(X.median())
            
            # Split train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scaling pour certains modèles
            self.scaler = RobustScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            self._log_message(f"✅ Données chargées: {len(self.X_train)} train, {len(self.X_test)} test")
            self._log_message(f"   Features: {self.X_train.shape[1]}, Classes: {len(np.unique(self.y_train))}")
            
            logger.info(f"Données chargées avec succès: {self.X_train.shape}")
            
        except Exception as e:
            error_msg = f"Erreur lors du chargement: {str(e)}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Erreur", error_msg)
    
    def _on_model_change(self, event=None):
        """Gère le changement de modèle sélectionné."""
        self.model_name = self.model_var.get()
        logger.info(f"Modèle sélectionné: {self.model_name}")
        
        # Vérification de disponibilité
        if self.model_name == "CNN" and not CNN_AVAILABLE:
            messagebox.showwarning("Attention", "PyTorch n'est pas installé. CNN non disponible.")
            return
        if self.model_name == "TabNet" and not TABNET_AVAILABLE:
            messagebox.showwarning("Attention", "TabNet n'est pas installé. TabNet non disponible.")
            return
    
    def _log_message(self, message: str):
        """Ajoute un message dans la zone de logs."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def _start_timer(self):
        """Démarre le timer."""
        self.start_time = time.perf_counter()
        self._update_timer()
    
    def _update_timer(self):
        """Met à jour l'affichage du timer."""
        if self.start_time and self.is_training:
            elapsed = time.perf_counter() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            
            timer_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
            self.timer_label.config(text=timer_str)
            
            # Détails par étape
            details = []
            for step, duration in self.timings.items():
                details.append(f"{step}: {duration:.2f}s")
            
            if details:
                self.timer_details.config(text=" | ".join(details))
            
            self.root.after(50, self._update_timer)
    
    def _start_training(self):
        """Démarre l'entraînement dans un thread séparé."""
        if self.data is None:
            messagebox.showwarning("Attention", "Veuillez charger des données d'abord.")
            return
        
        if self.is_training:
            messagebox.showinfo("Info", "Un entraînement est déjà en cours.")
            return
        
        self.model_name = self.model_var.get()
        self.is_training = True
        self.should_stop = False
        self.is_paused = False
        
        # Activation des boutons
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.NORMAL)
        
        # Reset
        self.timings = {}
        self.training_history = {'loss': [], 'accuracy': [], 'epoch': []}
        
        # Lancement dans un thread
        self.training_thread = threading.Thread(target=self._train_model, daemon=True)
        self.training_thread.start()
        
        self._start_timer()
    
    def _stop_training(self):
        """Arrête l'entraînement."""
        self.should_stop = True
        self.is_paused = False
        self._log_message("⏹ Arrêt demandé...")
    
    def _pause_training(self):
        """Met en pause/reprend l'entraînement."""
        if self.is_paused:
            self.is_paused = False
            self.pause_button.config(text="⏸ Pause")
            self._log_message("▶ Reprise de l'entraînement")
        else:
            self.is_paused = True
            self.pause_button.config(text="▶ Reprendre")
            self._log_message("⏸ Entraînement en pause...")
    
    def _wait_if_paused(self):
        """Attend si l'entraînement est en pause."""
        while self.is_paused and not self.should_stop:
            time.sleep(0.1)
    
    def _train_model(self):
        """Fonction principale d'entraînement (exécutée dans un thread)."""
        try:
            start_total = time.perf_counter()
            
            self._log_message(f"🚀 Début de l'entraînement: {self.model_name}")
            logger.info(f"Début entraînement: {self.model_name}")
            
            # === PRÉTRAITEMENT ===
            step_start = time.perf_counter()
            self._log_message("📊 Prétraitement des données...")
            self._wait_if_paused()
            
            # Sélection des features (limitation pour les modèles complexes)
            n_features = min(self.X_train.shape[1], 50) if self.model_name in ["CNN", "TabNet"] else self.X_train.shape[1]
            
            X_train_used = self.X_train_scaled[:, :n_features] if self.model_name in ["CNN", "TabNet", "Logistic Regression"] else self.X_train.values[:, :n_features]
            X_test_used = self.X_test_scaled[:, :n_features] if self.model_name in ["CNN", "TabNet", "Logistic Regression"] else self.X_test.values[:, :n_features]
            
            self.timings["Prétraitement"] = time.perf_counter() - step_start
            self._log_message(f"✅ Prétraitement terminé ({self.timings['Prétraitement']:.2f}s)")
            
            if self.should_stop:
                return
            
            # === ENTRAÎNEMENT ===
            step_start = time.perf_counter()
            self._log_message("🎯 Début entraînement du modèle...")
            self._wait_if_paused()
            
            if self.model_name == "Decision Tree":
                self.current_model = self._train_decision_tree(X_train_used, X_test_used)
            elif self.model_name == "Random Forest":
                self.current_model = self._train_random_forest(X_train_used, X_test_used)
            elif self.model_name == "Logistic Regression":
                self.current_model = self._train_logistic_regression(X_train_used, X_test_used)
            elif self.model_name == "CNN":
                self.current_model = self._train_cnn(X_train_used, X_test_used)
            elif self.model_name == "TabNet":
                self.current_model = self._train_tabnet(X_train_used, X_test_used)
            
            self.timings["Entraînement"] = time.perf_counter() - step_start
            self._log_message(f"✅ Entraînement terminé ({self.timings['Entraînement']:.2f}s)")
            
            if self.should_stop:
                return
            
            # === TUNING HYPERPARAMÈTRES ===
            step_start = time.perf_counter()
            self._log_message("🔧 Début tuning hyperparamètres...")
            self._wait_if_paused()
            
            best_params = self._tune_hyperparameters(X_train_used, X_test_used)
            
            self.timings["Tuning"] = time.perf_counter() - step_start
            self._log_message(f"✅ Tuning terminé ({self.timings['Tuning']:.2f}s)")
            
            if self.should_stop:
                return
            
            # === TESTING ===
            step_start = time.perf_counter()
            self._log_message("🧪 Début testing...")
            self._wait_if_paused()
            
            accuracy, f1 = self._test_model(X_test_used)
            
            self.timings["Testing"] = time.perf_counter() - step_start
            self._log_message(f"✅ Testing terminé ({self.timings['Testing']:.2f}s)")
            self._log_message(f"   📈 Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
            
            total_time = time.perf_counter() - start_total
            self._log_message(f"🎉 Pipeline complet terminé en {total_time:.2f}s")
            
            # Visualisation
            self.root.after(0, self._visualize_model)
            
        except Exception as e:
            error_msg = f"Erreur pendant l'entraînement: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_message(f"❌ {error_msg}")
            messagebox.showerror("Erreur", error_msg)
        finally:
            self.is_training = False
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.pause_button.config(state=tk.DISABLED, text="⏸ Pause"))
    
    def _train_decision_tree(self, X_train, X_test):
        """Entraîne un Decision Tree."""
        model = DecisionTreeClassifier(random_state=42, max_depth=10)
        model.fit(X_train, self.y_train)
        
        # Métriques pendant l'entraînement
        train_acc = model.score(X_train, self.y_train)
        test_acc = model.score(X_test, self.y_test)
        self._log_message(f"   Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return model
    
    def _train_random_forest(self, X_train, X_test):
        """Entraîne un Random Forest."""
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1, max_depth=10)
        model.fit(X_train, self.y_train)
        
        train_acc = model.score(X_train, self.y_train)
        test_acc = model.score(X_test, self.y_test)
        self._log_message(f"   Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Historique des arbres
        for i, tree in enumerate(model.estimators_[:5]):
            self.training_history['accuracy'].append(tree.score(X_test, self.y_test))
        
        return model
    
    def _train_logistic_regression(self, X_train, X_test):
        """Entraîne une Logistic Regression."""
        model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
        model.fit(X_train, self.y_train)
        
        train_acc = model.score(X_train, self.y_train)
        test_acc = model.score(X_test, self.y_test)
        self._log_message(f"   Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return model
    
    def _train_cnn(self, X_train, X_test):
        """Entraîne un CNN (MLP simple pour données tabulaires)."""
        if not CNN_AVAILABLE:
            raise ImportError("PyTorch non disponible")
        
        # Architecture simple
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        num_classes = len(np.unique(self.y_train))
        model = SimpleMLP(X_train.shape[1], num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Conversion en tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(self.y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Entraînement avec callbacks
        epochs = 10
        for epoch in range(epochs):
            if self.should_stop:
                break
            self._wait_if_paused()
            
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Métriques
            with torch.no_grad():
                train_pred = model(X_train_tensor).argmax(dim=1)
                train_acc = (train_pred == y_train_tensor).float().mean().item()
                
                test_pred = model(X_test_tensor).argmax(dim=1)
                test_acc = (test_pred == torch.LongTensor(self.y_test)).float().mean().item()
            
            self.training_history['loss'].append(loss.item())
            self.training_history['accuracy'].append(test_acc)
            self.training_history['epoch'].append(epoch + 1)
            
            self._log_message(f"   Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Test Acc={test_acc:.4f}")
            
            # Mise à jour graphique
            self.root.after(0, self._update_training_plot)
        
        return model
    
    def _train_tabnet(self, X_train, X_test):
        """Entraîne un TabNet."""
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet non disponible")
        
        model = TabNetClassifier(
            n_d=8, n_a=8,
            n_steps=3,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            seed=42,
            verbose=0
        )
        
        model.fit(
            X_train, self.y_train,
            eval_set=[(X_test, self.y_test)],
            eval_metric=['accuracy'],
            max_epochs=10,
            patience=5,
            batch_size=1024,
            virtual_batch_size=128
        )
        
        return model
    
    def _tune_hyperparameters(self, X_train, X_test):
        """Effectue une recherche d'hyperparamètres."""
        if self.should_stop:
            return {}
        
        # Paramètres à tester selon le modèle
        if self.model_name == "Decision Tree":
            param_grid = {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
            base_model = DecisionTreeClassifier(random_state=42)
        
        elif self.model_name == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=1)
        
        elif self.model_name == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear']
            }
            base_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
        
        else:
            self._log_message("   ⚠️ Tuning non implémenté pour ce modèle")
            return {}
        
        # Recherche randomisée (limité pour la démo)
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=5, cv=3, 
            scoring='accuracy', random_state=42, n_jobs=1
        )
        
        search.fit(X_train, self.y_train)
        
        best_params = search.best_params_
        best_score = search.best_score_
        
        self._log_message(f"   🏆 Meilleurs paramètres: {best_params}")
        self._log_message(f"   📊 Score CV: {best_score:.4f}")
        
        # Mise à jour du modèle avec les meilleurs paramètres
        self.current_model = search.best_estimator_
        
        return best_params
    
    def _test_model(self, X_test):
        """Teste le modèle et retourne les métriques."""
        if self.model_name in ["CNN"]:
            X_test_tensor = torch.FloatTensor(X_test)
            with torch.no_grad():
                predictions = self.current_model(X_test_tensor).argmax(dim=1).numpy()
        else:
            predictions = self.current_model.predict(X_test)
        
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        
        return accuracy, f1
    
    def _visualize_model(self):
        """Visualise la structure du modèle."""
        self.fig.clear()
        
        if self.model_name in ["Decision Tree", "Random Forest"]:
            self._visualize_tree()
        elif self.model_name == "Logistic Regression":
            self._visualize_logistic_regression()
        elif self.model_name == "CNN":
            self._visualize_cnn()
        elif self.model_name == "TabNet":
            self._visualize_tabnet()
        
        self.canvas.draw()
    
    def _visualize_tree(self):
        """Visualise un arbre de décision."""
        if DTreeViz_AVAILABLE and self.model_name == "Decision Tree":
            try:
                # Utilisation de dtreeviz pour visualiser l'arbre
                from dtreeviz.trees import dtreeviz
                viz = dtreeviz(
                    self.current_model,
                    self.X_train.values[:, :10] if self.X_train.shape[1] > 10 else self.X_train.values,
                    self.y_train,
                    target_name='label',
                    feature_names=[f'Feature_{i}' for i in range(min(10, self.X_train.shape[1]))],
                    class_names=[str(i) for i in np.unique(self.y_train)]
                )
                # Note: dtreeviz génère une image SVG, on affiche juste la structure ici
                self.ax = self.fig.add_subplot(111)
                self.ax.text(0.5, 0.5, "Visualisation dtreeviz disponible\n(Voir la sortie SVG)", 
                           ha='center', va='center', transform=self.ax.transAxes)
                self.ax.set_xticks([])
                self.ax.set_yticks([])
            except Exception as e:
                self._visualize_tree_structure()
        else:
            self._visualize_tree_structure()
        
        if self.model_name == "Random Forest":
            # Afficher les métriques des arbres
            self._visualize_random_forest_metrics()
    
    def _visualize_tree_structure(self):
        """Visualise la structure de l'arbre de manière simple."""
        from sklearn.tree import plot_tree
        
        self.ax = self.fig.add_subplot(111)
        
        if self.model_name == "Decision Tree":
            plot_tree(self.current_model, ax=self.ax, max_depth=3, filled=True, 
                     feature_names=[f'F{i}' for i in range(self.X_train.shape[1])],
                     class_names=[str(i) for i in np.unique(self.y_train)])
        else:
            # Pour Random Forest, afficher le premier arbre
            plot_tree(self.current_model.estimators_[0], ax=self.ax, max_depth=3, 
                     filled=True, feature_names=[f'F{i}' for i in range(self.X_train.shape[1])],
                     class_names=[str(i) for i in np.unique(self.y_train)])
        
        self.ax.set_title(f"Structure du {self.model_name}")
    
    def _visualize_random_forest_metrics(self):
        """Affiche les métriques du Random Forest."""
        if len(self.training_history['accuracy']) > 0:
            self.ax2 = self.fig.add_subplot(122)
            self.ax2.plot(range(len(self.training_history['accuracy'])), 
                         self.training_history['accuracy'], 'b-o')
            self.ax2.set_xlabel("Arbre #")
            self.ax2.set_ylabel("Accuracy")
            self.ax2.set_title("Accuracy par arbre")
            self.ax2.grid(True)
    
    def _visualize_logistic_regression(self):
        """Visualise les coefficients de la régression logistique."""
        self.ax = self.fig.add_subplot(111)
        
        coefficients = self.current_model.coef_[0][:20]  # Limite à 20 features
        feature_names = [f'F{i}' for i in range(len(coefficients))]
        
        colors = ['red' if c < 0 else 'green' for c in coefficients]
        self.ax.barh(feature_names, coefficients, color=colors)
        self.ax.set_xlabel("Coefficient")
        self.ax.set_title("Coefficients Logistic Regression (top 20)")
        self.ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        self.ax.grid(True, axis='x')
    
    def _visualize_cnn(self):
        """Visualise l'architecture et les métriques du CNN."""
        # Architecture
        self.ax = self.fig.add_subplot(121)
        layers = ['Input', 'FC1\n(128)', 'FC2\n(64)', 'Output']
        y_pos = np.arange(len(layers))
        
        self.ax.barh(y_pos, [1, 1, 1, 1], color=['blue', 'green', 'green', 'red'])
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(layers)
        self.ax.set_xlabel("Couches")
        self.ax.set_title("Architecture CNN")
        self.ax.set_xlim(0, 1.2)
        
        # Métriques d'entraînement
        if len(self.training_history['loss']) > 0:
            self.ax2 = self.fig.add_subplot(122)
            epochs = self.training_history['epoch']
            
            ax2_twin = self.ax2.twinx()
            line1 = self.ax2.plot(epochs, self.training_history['loss'], 'r-o', label='Loss')
            line2 = ax2_twin.plot(epochs, self.training_history['accuracy'], 'b-s', label='Accuracy')
            
            self.ax2.set_xlabel("Epoch")
            self.ax2.set_ylabel("Loss", color='r')
            ax2_twin.set_ylabel("Accuracy", color='b')
            self.ax2.set_title("Évolution Loss/Accuracy")
            self.ax2.grid(True)
            
            # Légende
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            self.ax2.legend(lines, labels, loc='upper left')
    
    def _visualize_tabnet(self):
        """Visualise les métriques du TabNet."""
        self.ax = self.fig.add_subplot(111)
        
        # TabNet stocke l'historique dans model.history
        if hasattr(self.current_model, 'history') and self.current_model.history:
            history = self.current_model.history
            
            if 'loss' in history and 'val_loss' in history:
                epochs = range(1, len(history['loss']) + 1)
                
                ax_twin = self.ax.twinx()
                line1 = self.ax.plot(epochs, history['loss'], 'r-o', label='Train Loss')
                line2 = self.ax.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
                line3 = ax_twin.plot(epochs, history['train_accuracy'], 'b-o', label='Train Acc')
                line4 = ax_twin.plot(epochs, history['val_accuracy'], 'b--s', label='Val Acc')
                
                self.ax.set_xlabel("Epoch")
                self.ax.set_ylabel("Loss", color='r')
                ax_twin.set_ylabel("Accuracy", color='b')
                self.ax.set_title("Évolution TabNet")
                self.ax.grid(True)
                
                lines = line1 + line2 + line3 + line4
                labels = [l.get_label() for l in lines]
                self.ax.legend(lines, labels, loc='upper left')
        else:
            self.ax.text(0.5, 0.5, "TabNet entraîné\n(Métriques non disponibles)", 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
    
    def _update_training_plot(self):
        """Met à jour le graphique pendant l'entraînement (pour CNN)."""
        if len(self.training_history['epoch']) > 0:
            self._visualize_cnn()


def main():
    """Point d'entrée principal."""
    root = tk.Tk()
    app = AlgorithmVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
