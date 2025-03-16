import sys
import numpy as np
import pandas as pd
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QFileDialog, QPushButton, QLabel, QComboBox, QHBoxLayout,
    QMessageBox, QInputDialog, QCheckBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting


# Helper class to visualize molecules
class MoleculeVisualizer:
    @staticmethod
    def plot_molecules(ax, molecule1_type, molecule2_type, distance, angle, line_length=2):
        """
        Plots Molecule 1 and Molecule 2 with fixed line lengths.
        """
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        # Coordinates for the key atom of Molecule 2
        x2 = distance * np.cos(angle_rad)
        y2 = distance * np.sin(angle_rad)

        # Plot Molecule 1
        ax.plot([0, -line_length], [0, 0], 'k-', lw=2)  # Line for Molecule 1
        ax.plot(0, 0, 'ro', markersize=10)  # Key atom at origin
        ax.text(-line_length, 0.2, molecule1_type, fontsize=12, ha='center')

        # Plot Molecule 2
        if distance != 0:
            u_x = x2 / distance
            u_y = y2 / distance
        else:
            u_x, u_y = 0, 0
        dx2 = x2 + line_length * u_x
        dy2 = y2 + line_length * u_y
        ax.plot([x2, dx2], [y2, dy2], 'b-', lw=2)
        ax.plot(x2, y2, 'go', markersize=10)
        ax.text(dx2, dy2 + 0.2, molecule2_type, fontsize=12, ha='center')

        # Draw line showing distance and annotate
        ax.plot([0, x2], [0, y2], 'r--', lw=1)
        ax.text(x2 / 2, y2 / 2 - 0.3, f'{distance:.2f} Å', fontsize=10, color='red', ha='center')

        # Annotate angle with an arc
        angle_arc = np.linspace(0, angle_rad, 100)
        arc_radius = distance / 4
        ax.plot(arc_radius * np.cos(angle_arc), arc_radius * np.sin(angle_arc), 'r-', lw=1)
        ax.text(arc_radius * np.cos(angle_rad / 2) + 0.1,
                arc_radius * np.sin(angle_rad / 2) + 0.1,
                f'{angle:.2f}°', fontsize=10, color='red')

        # Set fixed plot limits and labels
        ax.set_xlim(-3.5, 7.5)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_title('Molecular Interaction Visualization')
        ax.grid(True)


# Data exploration tab
class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        # Button to load dataset
        self.load_button = QPushButton('Load Dataset')
        self.load_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_button)

        self.data_info_label = QLabel('No dataset loaded.')
        self.layout.addWidget(self.data_info_label)

        # Plot options (Histogram, Scatter, 3D Scatter)
        self.plot_options_layout = QHBoxLayout()
        self.plot_label = QLabel('Select Plot Type:')
        self.plot_options_layout.addWidget(self.plot_label)
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(['Histogram', 'Scatter Plot', '3D Scatter Plot'])
        self.plot_options_layout.addWidget(self.plot_combo)
        self.plot_button = QPushButton('Plot Data')
        self.plot_button.clicked.connect(self.plot_data)
        self.plot_options_layout.addWidget(self.plot_button)
        self.layout.addLayout(self.plot_options_layout)

        # Save button for data plot
        self.save_data_plot_button = QPushButton('Save Data Plot')
        self.save_data_plot_button.clicked.connect(self.save_data_plot)
        self.layout.addWidget(self.save_data_plot_button)

        # Button to explore a single pair interaction
        self.explore_button = QPushButton('Explore Pair Interaction')
        self.explore_button.clicked.connect(self.explore_pair_interaction)
        self.layout.addWidget(self.explore_button)

        # Matplotlib figure and canvas for general data plots
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)

        # Matplotlib figure and canvas for pair interaction visualization
        self.pair_figure = Figure(figsize=(5, 4))
        self.pair_canvas = FigureCanvasQTAgg(self.pair_figure)
        self.layout.addWidget(self.pair_canvas)

        self.setLayout(self.layout)
        self.dataset = None

    def load_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_name:
            try:
                self.dataset = pd.read_csv(file_name)
                required_columns = ['Physical_Distance (Å)', 'Physical_Angle (°)', 'Energy', 'Molecule_1', 'Molecule_2']
                if not all(col in self.dataset.columns for col in required_columns):
                    raise ValueError(f"The dataset must contain the columns: {', '.join(required_columns)}")
                self.data_info_label.setText(f'Dataset loaded: {file_name}')
                QMessageBox.information(self, 'Success', 'Dataset loaded successfully.')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to load dataset: {e}')

    def plot_data(self):
        if self.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return
        plot_type = self.plot_combo.currentText()
        if plot_type == 'Histogram':
            self.plot_histogram()
        elif plot_type == 'Scatter Plot':
            self.plot_scatter()
        elif plot_type == '3D Scatter Plot':
            self.plot_3d_scatter()

    def plot_histogram(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        features = ['Physical_Distance (Å)', 'Physical_Angle (°)', 'Energy']
        self.dataset[features].hist(ax=ax)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_scatter(self):
        if self.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        features = ['Physical_Distance (Å)', 'Physical_Angle (°)', 'Energy']
        x_feature, ok = self.get_feature_selection('Select X-axis Feature', features)
        if not ok:
            return
        y_feature, ok = self.get_feature_selection('Select Y-axis Feature', features)
        if not ok:
            return
        scale_features = QMessageBox.question(
            self, 'Normalize Features',
            'Plot normalized features?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if scale_features == QMessageBox.Yes:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(self.dataset[features])
            scaled_df = pd.DataFrame(scaled_data, columns=features)
            x_data = scaled_df[x_feature]
            y_data = scaled_df[y_feature]
        else:
            x_data = self.dataset[x_feature]
            y_data = self.dataset[y_feature]
        ax.scatter(x_data, y_data, alpha=0.7)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        self.canvas.draw()

    def plot_3d_scatter(self):
        if self.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        scale_features = QMessageBox.question(
            self, 'Normalize Features',
            'Plot normalized features?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        features = ['Physical_Distance (Å)', 'Physical_Angle (°)', 'Energy']
        if scale_features == QMessageBox.Yes:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(self.dataset[features])
            scaled_df = pd.DataFrame(scaled_data, columns=features)
            x_data = scaled_df['Physical_Distance (Å)']
            y_data = scaled_df['Physical_Angle (°)']
            z_data = scaled_df['Energy']
        else:
            x_data = self.dataset['Physical_Distance (Å)']
            y_data = self.dataset['Physical_Angle (°)']
            z_data = self.dataset['Energy']
        ax.scatter(x_data, y_data, z_data, alpha=0.7)
        ax.set_xlabel('Physical_Distance (Å)')
        ax.set_ylabel('Physical_Angle (°)')
        ax.set_zlabel('Energy')
        self.canvas.draw()

    def get_feature_selection(self, title, features):
        feature, ok = QInputDialog.getItem(self, title, 'Feature:', features, 0, False)
        return feature, ok

    def explore_pair_interaction(self):
        if self.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return
        random_row = self.dataset.sample(n=1).iloc[0]
        molecule1_type = random_row['Molecule_1']
        molecule2_type = random_row['Molecule_2']
        distance = random_row['Physical_Distance (Å)']
        angle = random_row['Physical_Angle (°)']
        self.pair_figure.clear()
        ax = self.pair_figure.add_subplot(111)
        MoleculeVisualizer.plot_molecules(ax, molecule1_type, molecule2_type, distance, angle)
        self.pair_canvas.draw()

    def save_data_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data Plot", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            self.figure.savefig(file_path)
            QMessageBox.information(self, "Plot Saved", f"Data plot saved as {file_path}")


# ModelTab for running clustering and displaying random molecule interactions
class ModelTab(QWidget):
    def __init__(self, data_tab):
        super().__init__()
        self.data_tab = data_tab
        self.layout = QVBoxLayout()
        self.algorithm_label = QLabel('Select Clustering Algorithm:')
        self.layout.addWidget(self.algorithm_label)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['K-means', 'Hierarchical Clustering'])
        self.layout.addWidget(self.algorithm_combo)
        self.params_layout = QHBoxLayout()
        self.num_clusters_label = QLabel('Number of Clusters (k):')
        self.params_layout.addWidget(self.num_clusters_label)
        self.num_clusters_combo = QComboBox()
        self.num_clusters_combo.addItems([str(i) for i in range(2, 11)])
        self.params_layout.addWidget(self.num_clusters_combo)
        self.layout.addLayout(self.params_layout)
        self.scaling_checkbox = QCheckBox('Normalize Features')
        self.scaling_checkbox.setChecked(True)
        self.layout.addWidget(self.scaling_checkbox)
        self.run_button = QPushButton('Run Clustering')
        self.run_button.clicked.connect(self.run_clustering)
        self.layout.addWidget(self.run_button)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def run_clustering(self):
        if self.data_tab.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return
        self.molecule_indices = random.sample(range(len(self.data_tab.dataset)), 100)
        self.current_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_random_molecules)
        self.timer.start(50)  # 50 milliseconds

    def display_random_molecules(self):
        if self.current_index < len(self.molecule_indices):
            idx = self.molecule_indices[self.current_index]
            row = self.data_tab.dataset.iloc[idx]
            molecule1_type = row['Molecule_1']
            molecule2_type = row['Molecule_2']
            distance = row['Physical_Distance (Å)']
            angle = row['Physical_Angle (°)']
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            MoleculeVisualizer.plot_molecules(ax, molecule1_type, molecule2_type, distance, angle)
            self.canvas.draw()
            self.current_index += 1
        else:
            self.timer.stop()
            self.perform_clustering()

    def perform_clustering(self):
        algorithm = self.algorithm_combo.currentText()
        n_clusters = int(self.num_clusters_combo.currentText())
        features = self.data_tab.dataset[['Physical_Distance (Å)', 'Physical_Angle (°)', 'Energy']]
        if self.scaling_checkbox.isChecked():
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)
        else:
            features_scaled = features.values
        if algorithm == 'K-means':
            model = KMeans(n_clusters=n_clusters)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)
        self.data_tab.dataset['Cluster'] = model.fit_predict(features_scaled)
        QMessageBox.information(self, 'Success', 'Clustering completed successfully.')


# ResultsTab for visualizing clustering results and saving plots
class ResultsTab(QWidget):
    def __init__(self, data_tab):
        super().__init__()
        self.data_tab = data_tab
        self.layout = QVBoxLayout()
        # Upper layout: 3D cluster plot and heatmap
        self.upper_layout = QHBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.upper_layout.addWidget(self.canvas)
        self.heatmap_figure = Figure()
        self.heatmap_canvas = FigureCanvasQTAgg(self.heatmap_figure)
        self.upper_layout.addWidget(self.heatmap_canvas)
        self.layout.addLayout(self.upper_layout)
        # Plot options
        self.plot_options_layout = QHBoxLayout()
        self.color_by_label = QLabel('Color by:')
        self.plot_options_layout.addWidget(self.color_by_label)
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(['Cluster', 'Molecule Pair'])
        self.plot_options_layout.addWidget(self.color_by_combo)
        self.plot_button = QPushButton('Plot Clusters')
        self.plot_button.clicked.connect(self.plot_clusters)
        self.plot_options_layout.addWidget(self.plot_button)
        self.layout.addLayout(self.plot_options_layout)
        # Save buttons for cluster plot and heatmap
        self.save_cluster_plot_button = QPushButton("Save Cluster Plot")
        self.save_cluster_plot_button.clicked.connect(self.save_cluster_plot)
        self.layout.addWidget(self.save_cluster_plot_button)
        self.save_heatmap_button = QPushButton("Save Heatmap")
        self.save_heatmap_button.clicked.connect(self.save_heatmap)
        self.layout.addWidget(self.save_heatmap_button)
        # Cluster selection
        self.cluster_list_label = QLabel('Select a Cluster to Analyze:')
        self.layout.addWidget(self.cluster_list_label)
        self.cluster_list_widget = QListWidget()
        self.cluster_list_widget.itemClicked.connect(self.display_cluster_histogram)
        self.layout.addWidget(self.cluster_list_widget)
        # Show Cluster Map button
        self.show_cluster_map_button = QPushButton('Show Cluster Map')
        self.show_cluster_map_button.clicked.connect(self.show_cluster_map)
        self.layout.addWidget(self.show_cluster_map_button)
        # Histogram area
        self.histogram_label = QLabel('Molecule Pair Distribution in Selected Cluster:')
        self.histogram_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.layout.addWidget(self.histogram_label)
        self.histogram_figure = Figure()
        self.histogram_canvas = FigureCanvasQTAgg(self.histogram_figure)
        self.layout.addWidget(self.histogram_canvas)
        self.setLayout(self.layout)

    def plot_clusters(self):
        if self.data_tab.dataset is None or 'Cluster' not in self.data_tab.dataset.columns:
            QMessageBox.warning(self, 'Error', 'Please run clustering first.')
            return
        color_by = self.color_by_combo.currentText()
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        data = self.data_tab.dataset.copy()
        if color_by == 'Cluster':
            groups = data['Cluster'].unique()
            group_field = 'Cluster'
        else:
            data['Molecule Pair'] = data['Molecule_1'] + ' & ' + data['Molecule_2']
            groups = data['Molecule Pair'].unique()
            group_field = 'Molecule Pair'
        cmap = cm.get_cmap('viridis', len(groups))
        group_to_idx = {group: idx for idx, group in enumerate(groups)}
        for group in groups:
            idx = group_to_idx[group]
            group_data = data[data[group_field] == group]
            ax.scatter(
                group_data['Physical_Distance (Å)'],
                group_data['Physical_Angle (°)'],
                group_data['Energy'],
                color=cmap(idx)
            )
        ax.set_xlabel('Physical Distance (Å)')
        ax.set_ylabel('Physical Angle (°)')
        ax.set_zlabel('Energy')
        ax.legend()
        self.canvas.draw()
        self.heatmap_figure.clear()
        self.heatmap_canvas.draw()
        self.update_cluster_list()

    def update_cluster_list(self):
        self.cluster_list_widget.clear()
        clusters = sorted(self.data_tab.dataset['Cluster'].unique())
        for cluster in clusters:
            item = QListWidgetItem(f'Cluster {cluster}')
            item.setData(Qt.UserRole, cluster)
            self.cluster_list_widget.addItem(item)

    def display_cluster_histogram(self, item):
        self.selected_cluster = item.data(Qt.UserRole)
        data = self.data_tab.dataset
        cluster_data = data[data['Cluster'] == self.selected_cluster].copy()
        cluster_data['Molecule Pair'] = cluster_data['Molecule_1'] + ' & ' + cluster_data['Molecule_2']
        pair_counts = cluster_data['Molecule Pair'].value_counts().sort_index()
        self.histogram_figure.clear()
        ax = self.histogram_figure.add_subplot(111)
        pair_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Molecule Pair')
        ax.set_ylabel('Count')
        ax.set_title(f'Molecule Pair Distribution in Cluster {self.selected_cluster}')
        self.histogram_figure.tight_layout()
        self.histogram_canvas.draw()

    def show_cluster_map(self):
        if not hasattr(self, 'selected_cluster'):
            QMessageBox.warning(self, 'Error', 'Please select a cluster first.')
            return
        cluster_data = self.data_tab.dataset[self.data_tab.dataset['Cluster'] == self.selected_cluster]
        if cluster_data.empty:
            QMessageBox.warning(self, 'Error', 'Selected cluster is empty.')
            return
        self.cluster_indices = cluster_data.index.tolist()
        if len(self.cluster_indices) < 100:
            self.molecule_indices = self.cluster_indices
        else:
            self.molecule_indices = random.sample(self.cluster_indices, 100)
        self.current_index = 0
        self.x_positions = []
        self.y_positions = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_cluster_molecules)
        self.timer.start(50)

    def display_cluster_molecules(self):
        if self.current_index < len(self.molecule_indices):
            idx = self.molecule_indices[self.current_index]
            row = self.data_tab.dataset.loc[idx]
            molecule1_type = row['Molecule_1']
            molecule2_type = row['Molecule_2']
            distance = row['Physical_Distance (Å)']
            angle = row['Physical_Angle (°)']
            self.heatmap_figure.clear()
            ax = self.heatmap_figure.add_subplot(111)
            MoleculeVisualizer.plot_molecules(ax, molecule1_type, molecule2_type, distance, angle)
            angle_rad = np.deg2rad(angle)
            x2 = distance * np.cos(angle_rad)
            y2 = distance * np.sin(angle_rad)
            self.x_positions.append(x2)
            self.y_positions.append(y2)
            self.heatmap_canvas.draw()
            self.current_index += 1
        else:
            self.timer.stop()
            self.display_heatmap()

    def display_heatmap(self):
        self.heatmap_figure.clear()
        ax = self.heatmap_figure.add_subplot(111)
        cluster_data = self.data_tab.dataset[self.data_tab.dataset['Cluster'] == self.selected_cluster]
        x_positions = []
        y_positions = []
        for _, row in cluster_data.iterrows():
            distance = row['Physical_Distance (Å)']
            angle = row['Physical_Angle (°)']
            angle_rad = np.deg2rad(angle)
            x2 = distance * np.cos(angle_rad)
            y2 = distance * np.sin(angle_rad)
            x_positions.append(x2)
            y_positions.append(y2)
        heatmap, xedges, yedges = np.histogram2d(
            x_positions, y_positions, bins=50, range=[[-3.5, 7.5], [-6, 6]]
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto'
        )
        self.heatmap_figure.colorbar(im, ax=ax, label='Frequency')
        ax.plot(0, 0, 'wo', markersize=10, label='Molecule 1 Key Atom')
        ax.plot([0, -2], [0, 0], 'w-', lw=2, label='Molecule 1')
        ax.set_title(f'Heatmap of Molecule 2 Key Atom Positions in Cluster {self.selected_cluster}')
        ax.set_xlabel('X Position (Å)')
        ax.set_ylabel('Y Position (Å)')
        ax.set_aspect('equal', 'box')
        self.heatmap_canvas.draw()

    def save_cluster_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Cluster Plot", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            self.figure.savefig(file_path)
            QMessageBox.information(self, "Plot Saved", f"Cluster plot saved as {file_path}")

    def save_heatmap(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Heatmap", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            self.heatmap_figure.savefig(file_path)
            QMessageBox.information(self, "Plot Saved", f"Heatmap saved as {file_path}")


# Main application window that includes all tabs
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Molecule Interaction Clustering')
        self.setGeometry(50, 50, 1024, 768)
        self.tabs = QTabWidget()
        self.data_tab = DataTab()
        self.model_tab = ModelTab(self.data_tab)
        self.results_tab = ResultsTab(self.data_tab)
        self.tabs.addTab(self.data_tab, 'Data')
        self.tabs.addTab(self.model_tab, 'Model')
        self.tabs.addTab(self.results_tab, 'Results')
        self.setCentralWidget(self.tabs)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

