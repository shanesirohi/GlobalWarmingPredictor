import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QSlider, QPushButton, QMessageBox, QPlainTextEdit
from PyQt5.QtCore import Qt


class GlobalWarmingEstimator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Global Warming Estimator')
        self.layout = QVBoxLayout()

        # Create sliders for various factors
        self.factor_labels = ['Fossil Fuel Usage:', 'Transportation:', 'Energy Consumption:', 'Deforestation:', 'Waste Management:', 'Industrial Processes:']
        self.factor_sliders = []
        self.factor_value_labels = []

        for label in self.factor_labels:
            factor_label = QLabel(label)
            factor_slider = QSlider(Qt.Horizontal)
            factor_slider.setMinimum(0)
            factor_slider.setMaximum(100)
            factor_slider.setValue(50)
            factor_slider.setTickInterval(10)
            factor_slider.setTickPosition(QSlider.TicksBelow)
            factor_slider.valueChanged.connect(self.update_slider_value)
            self.layout.addWidget(factor_label)
            self.layout.addWidget(factor_slider)
            self.factor_sliders.append(factor_slider)

            factor_value_label = QLabel('50')  # Initial value
            self.layout.addWidget(factor_value_label)
            self.factor_value_labels.append(factor_value_label)

        self.calculate_button = QPushButton('Calculate')
        self.calculate_button.clicked.connect(self.calculate_effects)
        self.layout.addWidget(self.calculate_button)

        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        self.setLayout(self.layout)

    def update_slider_value(self):
        sender = self.sender()
        index = self.factor_sliders.index(sender)
        self.factor_value_labels[index].setText(str(sender.value()))

    def calculate_effects(self):
        # Set weights for each factor (adjust as needed)
        weights = [0.2, 0.15, 0.15, 0.15, 0.15, 0.2]

        # Gather values from sliders for each factor
        factor_values = [slider.value() for slider in self.factor_sliders]

        # Calculate estimated effects based on factors and weights
        estimated_effects = [weight * value for weight, value in zip(weights, factor_values)]

        # Determine the effect levels based on the estimated effects
        effect_levels = []
        for effect in estimated_effects:
            if effect < 20:
                effect_levels.append('Very Low')
            elif effect < 40:
                effect_levels.append('Low')
            elif effect < 60:
                effect_levels.append('Moderate')
            elif effect < 80:
                effect_levels.append('High')
            else:
                effect_levels.append('Very High')

        # Display estimation results
        message = '\n'.join([f'{label}: {level}' for label, level in zip(self.factor_labels, effect_levels)])
        self.output_text.setPlainText(message)

        # Plot multiple graphs for estimated effects and temperature
        self.plot_graphs(self.factor_labels, factor_values, estimated_effects, weights)

        # Calculate and display solutions
        solutions = self.get_solutions(self.factor_labels, factor_values, weights)
        self.output_text.appendPlainText('\nSuggested Solutions:')
        for factor, reduction_score in solutions.items():
            self.output_text.appendPlainText(f'- Reduce {factor}: {reduction_score*100:.2f}%')

    def plot_graphs(self, factor_labels, factor_values, estimated_effects, weights):
        num_factors = len(factor_labels)
        num_rows = (num_factors + 1) // 2
        num_cols = 2

        colors = ['#FF5733', '#FFC300', '#C70039', '#900C3F', '#581845', '#0A1172']

        plt.figure(figsize=(12, 8))

        for i, factor_value in enumerate(factor_values):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.bar(range(num_factors), factor_value, color=colors[i], alpha=0.7)
            plt.xticks(range(num_factors), factor_labels, rotation=45)
            plt.ylabel('Values')
            plt.title(f'{factor_labels[i]}')

        plt.tight_layout()

        plt.figure(figsize=(8, 6))
        plt.plot(factor_labels, [0] * num_factors, color='black', linewidth=1, linestyle='--', alpha=0.5)
        plt.plot(factor_labels, estimated_effects, color='red', linewidth=2)
        plt.fill_between(factor_labels, [0] * num_factors, estimated_effects, color='red', alpha=0.2)
        plt.xticks(rotation=45)
        plt.ylabel('Estimated Effects')
        plt.title('Estimated Effects of Global Warming')

        for i, effect in enumerate(estimated_effects):
            plt.text(i, effect, f'{effect:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        plt.figure(figsize=(18, 16))

        plt.subplot(3, 3, 4)
        plt.pie(factor_values, labels=factor_labels, autopct='%1.1f%%', colors=colors)
        plt.title('Factor Values')

        plt.subplot(3, 3, 5)
        bottom = [0] * num_factors
        for i, effect in enumerate(estimated_effects):
            plt.bar(0, effect, bottom=bottom[i], color=colors[i], alpha=0.7)
            bottom[i] += effect
        plt.xticks([])
        plt.ylabel('Estimated Effects')
        plt.title('Stacked Estimated Effects')

        basic_temperature = [15, 16, 17, 18, 19, 20]
        plt.subplot(3, 3, 6)
        plt.plot(basic_temperature, estimated_effects, color='green', linewidth=2)
        plt.xlabel('Basic Temperature (°C)')
        plt.ylabel('Estimated Effects')
        plt.title('Estimated Effects vs Basic Temperature')

        predicted_temperature = self.predict_temperature(factor_values)
        plt.subplot(3, 3, 1)
        plt.bar('Predicted', predicted_temperature, color='blue')
        plt.title('Predicted Temperature')
        plt.ylabel('Temperature (°C)')

        common_problems = self.get_common_problems(factor_values)
        plt.subplot(3, 3, 2)
        plt.barh(range(len(common_problems)), common_problems.values(), align='center', color='orange')
        plt.yticks(range(len(common_problems)), common_problems.keys())
        plt.title('Common Problems')
        plt.xlabel('Effect Level')

        plt.subplot(3, 3, 3)
        factor_contributions = [value * weight for value, weight in zip(factor_values, weights)]
        plt.barh(factor_labels, factor_contributions, color=colors)
        plt.title('Contribution of Factors')
        plt.xlabel('Contribution')
        plt.ylabel('Factors')

        plt.subplot(3, 3, 7)
        solutions = self.get_solutions(factor_labels, factor_values, weights)
        plt.barh(range(len(solutions)), solutions.values(), align='center', color='purple')
        plt.yticks(range(len(solutions)), solutions.keys())
        plt.title('Solutions for Reducing Impact')
        plt.xlabel('Score')
        plt.ylabel('Factors')

        plt.tight_layout()
        plt.show()

    def predict_temperature(self, factor_values):
        # A simple model to predict temperature based on factor values
        temperature = sum(factor_values) / len(factor_values) + 15
        return temperature

    def get_common_problems(self, factor_values):
        # Assign common problems based on the factor values
        common_problems = {
            'Rising Sea Levels': factor_values[0],
            'Increased Extreme Weather Events': factor_values[1],
            'Melting Ice Caps': factor_values[2],
            'Loss of Biodiversity': factor_values[3],
            'Health Risks': factor_values[4],
            'Food Shortages': factor_values[5]
        }
        return common_problems

    def get_solutions(self, factor_labels, factor_values, weights):
        # Calculate the solutions based on the contribution and weights
        total_contribution = sum([value * weight for value, weight in zip(factor_values, weights)])
        solutions = {}
        for label, value, weight in zip(factor_labels, factor_values, weights):
            contribution = value * weight
            reduction_score = (total_contribution - contribution) / total_contribution
            solutions[label] = reduction_score
        return solutions


if __name__ == '__main__':
    app = QApplication(sys.argv)
    estimator = GlobalWarmingEstimator()
    estimator.show()
    sys.exit(app.exec_())
