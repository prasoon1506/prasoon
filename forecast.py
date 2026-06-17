import pandas as pd
import numpy as np
import warnings
import logging
import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from darts import TimeSeries
from darts.models import TFTModel, RNNModel

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").propagate = False
logging.getLogger("cmdstanpy").handlers = []
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)

class EndogenousRecurrentNetworkNode(nn.Module):
    def __init__(self, input_dimensionality=1, hidden_node_capacity=128, layer_depth=3):
        super(EndogenousRecurrentNetworkNode, self).__init__()
        self.h_dim = hidden_node_capacity
        self.l_depth = layer_depth
        self.lstm_architecture = nn.LSTM(input_dimensionality, hidden_node_capacity, layer_depth, batch_first=True, dropout=0.2)
        self.regression_head_topology = nn.Sequential(
            nn.Linear(hidden_node_capacity, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_tensor_matrix):
        hidden_initialization_zero = torch.zeros(self.l_depth, input_tensor_matrix.size(0), self.h_dim).requires_grad_()
        cell_state_initialization_zero = torch.zeros(self.l_depth, input_tensor_matrix.size(0), self.h_dim).requires_grad_()
        recurrent_output_sequence, (hidden_state_n, cell_state_n) = self.lstm_architecture(
            input_tensor_matrix, 
            (hidden_initialization_zero.detach(), cell_state_initialization_zero.detach())
        )
        final_projection_output = self.regression_head_topology(recurrent_output_sequence[:, -1, :]) 
        return final_projection_output

class MathematicalSubstrateInterface:
    def __init__(self, string_identifier):
        self.algorithm_signature = string_identifier
    def _execute_tensor_ingestion_and_optimization(self, training_dataframe_matrix, exogenous_tensor_matrix=None): pass
    def _execute_extrapolation_projection(self, target_step_count, exogenous_tensor_matrix=None): pass

class ProphetOptimizationEngine(MathematicalSubstrateInterface):
    def __init__(self):
        super().__init__("Facebook_Prophet_MAP_Estimation")
        self._compiled_model_object = None

    def _execute_tensor_ingestion_and_optimization(self, training_dataframe_matrix, exogenous_tensor_matrix=None):
        self._compiled_model_object = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        if exogenous_tensor_matrix is not None:
            for covariate_column_identifier in exogenous_tensor_matrix.columns:
                if covariate_column_identifier not in ['ds', 'y']: 
                    self._compiled_model_object.add_regressor(covariate_column_identifier)
        
        parsed_training_matrix = training_dataframe_matrix.rename(columns={'t_node': 'ds', 'q_val': 'y'})
        if exogenous_tensor_matrix is not None:
            parsed_training_matrix = pd.merge(parsed_training_matrix, exogenous_tensor_matrix, on='ds', how='left').fillna(0)
            
        self._compiled_model_object.fit(parsed_training_matrix)

    def _execute_extrapolation_projection(self, target_step_count, exogenous_tensor_matrix=None):
        future_dataframe_topology = self._compiled_model_object.make_future_dataframe(periods=target_step_count, freq='MS')
        if exogenous_tensor_matrix is not None:
            future_dataframe_topology = pd.merge(future_dataframe_topology, exogenous_tensor_matrix, on='ds', how='left').fillna(0)
        extrapolated_vector = self._compiled_model_object.predict(future_dataframe_topology)['yhat'].tail(target_step_count).values
        return np.maximum(0, extrapolated_vector)

class HoltWintersExponentialEngine(MathematicalSubstrateInterface):
    def __init__(self):
        super().__init__("Holt_Winters_Statistical_Smoothing")
        self._compiled_model_object = None
        self._fallback_baseline_metric = 0

    def _execute_tensor_ingestion_and_optimization(self, training_dataframe_matrix, exogenous_tensor_matrix=None):
        time_series_vector_array = training_dataframe_matrix['q_val'].values
        try:
            self._compiled_model_object = ExponentialSmoothing(
                time_series_vector_array, 
                trend='add', 
                seasonal=None, 
                initialization_method="estimated"
            ).fit(optimized=True)
        except:
            self._compiled_model_object = None
            self._fallback_baseline_metric = np.mean(time_series_vector_array[-6:]) if len(time_series_vector_array) > 0 else 0

    def _execute_extrapolation_projection(self, target_step_count, exogenous_tensor_matrix=None):
        if self._compiled_model_object is None:
            return np.array([self._fallback_baseline_metric] * target_step_count)
        return np.maximum(0, self._compiled_model_object.forecast(target_step_count))

class DeepLearningDartsOrchestrator(MathematicalSubstrateInterface):
    def __init__(self, architecture_classification="TFT"):
        super().__init__(f"SOTA_Darts_{architecture_classification}")
        self.architecture_classification = architecture_classification
        self._compiled_model_object = None
        self._internal_time_series_representation = None

    def _execute_tensor_ingestion_and_optimization(self, training_dataframe_matrix, exogenous_tensor_matrix=None):
        sorted_dataframe_matrix = training_dataframe_matrix.sort_values('t_node')
        self._internal_time_series_representation = TimeSeries.from_dataframe(
            sorted_dataframe_matrix, time_col='t_node', value_cols='q_val', fill_missing_dates=True, freq='MS'
        )
        
        compiled_covariates_representation = None
        if exogenous_tensor_matrix is not None:
            sorted_exogenous_matrix = exogenous_tensor_matrix.sort_values('ds')
            compiled_covariates_representation = TimeSeries.from_dataframe(
                sorted_exogenous_matrix, time_col='ds', fill_missing_dates=True, freq='MS'
            )

        if self.architecture_classification == "TFT":
            self._compiled_model_object = TFTModel(
                input_chunk_length=6, output_chunk_length=1, hidden_size=64, 
                lstm_layers=2, num_attention_heads=4, n_epochs=200, dropout=0.1, random_state=42
            )
        else:
            self._compiled_model_object = RNNModel(
                model='LSTM', input_chunk_length=6, hidden_dim=64, 
                n_rnn_layers=3, n_epochs=200, dropout=0.1, random_state=42
            )
            
        try:
            if compiled_covariates_representation is not None:
                self._compiled_model_object.fit(self._internal_time_series_representation, future_covariates=compiled_covariates_representation, verbose=False)
            else:
                self._compiled_model_object.fit(self._internal_time_series_representation, verbose=False)
        except Exception:
            self._compiled_model_object = None 

    def _execute_extrapolation_projection(self, target_step_count, exogenous_tensor_matrix=None):
        if self._compiled_model_object is None: return np.zeros(target_step_count)
        try:
            prediction_object = self._compiled_model_object.predict(n=target_step_count)
            return np.maximum(0, prediction_object.values().flatten())
        except:
            return np.zeros(target_step_count)

class MasterExecutionManifold:
    def __init__(self, input_filepath_string='dealer_data.xlsx'):
        self._source_filepath = input_filepath_string
        self._execute_initial_tensor_synthesis()

    def _execute_initial_tensor_synthesis(self):
        dataframe_sales_raw = pd.read_excel(self._source_filepath, sheet_name=0)
        dataframe_price_raw = pd.read_excel(self._source_filepath, sheet_name=1)
        
        self.meta_dimensional_frame = dataframe_sales_raw.iloc[:, :3].drop_duplicates()
        
        temporal_column_identifiers = dataframe_sales_raw.columns[3:]
        try:
            temporal_index_object = pd.to_datetime(temporal_column_identifiers, format='%b-%y')
        except:
            temporal_index_object = pd.to_datetime(temporal_column_identifiers)
            
        self.t_omega_boundary = temporal_index_object.max()
        self.backtest_validation_domain = pd.date_range(end=self.t_omega_boundary, periods=6, freq='MS')
        self.target_forecast_domain = pd.date_range(start=self.t_omega_boundary + pd.DateOffset(months=1), periods=1, freq='MS')
        
        self.df_sales_melted = dataframe_sales_raw.melt(id_vars=dataframe_sales_raw.columns[:3], var_name='t_str', value_name='q_val')
        self.df_sales_melted['t_node'] = pd.to_datetime(self.df_sales_melted['t_str'], errors='coerce')
        
        self.df_price_melted = dataframe_price_raw.melt(id_vars=['State'], var_name='t_str', value_name='price')
        self.df_price_melted['ds'] = pd.to_datetime(self.df_price_melted['t_str'], errors='coerce')
        
    def _construct_exogenous_state_matrix(self, temporal_history_frame, geographic_state_identifier, exclusivity_flag_identifier):
        temporal_origin_point = temporal_history_frame['t_node'].min()
        absolute_future_boundary = self.target_forecast_domain[-1]
        
        unbroken_temporal_continuum = pd.date_range(start=temporal_origin_point, end=absolute_future_boundary, freq='MS')
        baseline_exogenous_structure = pd.DataFrame({'ds': unbroken_temporal_continuum})
        
        filtered_geographic_pricing = self.df_price_melted[(self.df_price_melted['State'] == geographic_state_identifier)][['ds', 'price']].copy()
        
        unified_exogenous_tensor = pd.merge(baseline_exogenous_structure, filtered_geographic_pricing, on='ds', how='left')
        unified_exogenous_tensor['price'] = unified_exogenous_tensor['price'].bfill().ffill().fillna(300) 
        unified_exogenous_tensor['is_exclusive'] = 1 if str(exclusivity_flag_identifier).strip().upper() == 'YES' else 0
        
        return unified_exogenous_tensor

    def execute_global_matrix_tournament(self):
        unique_dealer_identifiers = self.df_sales_melted['Dealer_Code'].unique()
        compiled_output_dictionary_list = []
        
        process_tracker_object = tqdm(total=len(unique_dealer_identifiers), desc="Processing Neural Manifolds", unit="node", dynamic_ncols=True)
        
        for singular_dealer_id in unique_dealer_identifiers:
            process_tracker_object.set_postfix_str(f"ID: {singular_dealer_id}")
            historical_subset_frame = self.df_sales_melted[self.df_sales_melted['Dealer_Code'] == singular_dealer_id].sort_values('t_node')
            
            if len(historical_subset_frame) < 12: 
                process_tracker_object.update(1)
                continue 
            
            extracted_state_parameter = historical_subset_frame['State'].iloc[0]
            extracted_exclusivity_parameter = historical_subset_frame['Exclusive'].iloc[0]
            
            constructed_exogenous_tensor = self._construct_exogenous_state_matrix(historical_subset_frame, extracted_state_parameter, extracted_exclusivity_parameter)
            
            algorithm_architectures = {
                'Holt_Winters': HoltWintersExponentialEngine(),
                'Prophet': ProphetOptimizationEngine(),
                'TFT': DeepLearningDartsOrchestrator("TFT"),
                'DeepAR': DeepLearningDartsOrchestrator("DeepAR")
            }
            
            rolling_validation_error_ledger = {alg: [] for alg in list(algorithm_architectures.keys()) + ['MLP', 'Custom_Ensemble', 'Short_History_RF', 'Sparse_Poisson', 'LSTM']}
            rolling_validation_prediction_ledger = {alg: [] for alg in rolling_validation_error_ledger.keys()}
            actual_validation_ground_truths = []

            for chronological_validation_step in self.backtest_validation_domain:
                iterative_training_boundary = historical_subset_frame[historical_subset_frame['t_node'] < chronological_validation_step]
                iterative_ground_truth_value = historical_subset_frame[historical_subset_frame['t_node'] == chronological_validation_step]['q_val'].values
                
                if len(iterative_training_boundary) < 6 or len(iterative_ground_truth_value) == 0:
                    actual_validation_ground_truths.append(0)
                    for k in rolling_validation_prediction_ledger.keys():
                        rolling_validation_prediction_ledger[k].append(0)
                    continue
                    
                actual_validation_ground_truths.append(iterative_ground_truth_value[0])
                
                for algo_name, algo_object in algorithm_architectures.items():
                    try:
                        algo_object._execute_tensor_ingestion_and_optimization(iterative_training_boundary, constructed_exogenous_tensor)
                        projected_step_value = algo_object._execute_extrapolation_projection(1, constructed_exogenous_tensor)
                        rolling_validation_prediction_ledger[algo_name].append(projected_step_value[0])
                    except:
                        rolling_validation_prediction_ledger[algo_name].append(0)

                x_dimensional_training_vector = np.arange(len(iterative_training_boundary)).reshape(-1, 1)
                y_dimensional_training_vector = iterative_training_boundary['q_val'].values
                x_dimensional_validation_step = np.array([[len(iterative_training_boundary)]])
                
                scalar_transformation_object = StandardScaler()
                x_dimensional_training_vector_scaled = scalar_transformation_object.fit_transform(x_dimensional_training_vector)
                x_dimensional_validation_step_scaled = scalar_transformation_object.transform(x_dimensional_validation_step)
                
                mlp_architecture_object = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=1500, early_stopping=True).fit(x_dimensional_training_vector_scaled, y_dimensional_training_vector)
                mlp_projected_value = np.maximum(0, mlp_architecture_object.predict(x_dimensional_validation_step_scaled))
                rolling_validation_prediction_ledger['MLP'].append(mlp_projected_value[0])

                huber_architecture_object = HuberRegressor(epsilon=1.35, max_iter=1500).fit(x_dimensional_training_vector_scaled, y_dimensional_training_vector)
                huber_projected_value = np.maximum(0, huber_architecture_object.predict(x_dimensional_validation_step_scaled))
                ensemble_projected_value = (mlp_projected_value[0] * 0.5) + (huber_projected_value[0] * 0.5)
                rolling_validation_prediction_ledger['Custom_Ensemble'].append(ensemble_projected_value)
                
                rf_architecture_object = RandomForestRegressor(n_estimators=100, random_state=42).fit(x_dimensional_training_vector_scaled, y_dimensional_training_vector)
                rf_projected_value = np.maximum(0, rf_architecture_object.predict(x_dimensional_validation_step_scaled))
                rolling_validation_prediction_ledger['Short_History_RF'].append(rf_projected_value[0])
                
                poisson_architecture_object = PoissonRegressor(max_iter=1000).fit(x_dimensional_training_vector_scaled, y_dimensional_training_vector)
                poisson_projected_value = np.maximum(0, poisson_architecture_object.predict(x_dimensional_validation_step_scaled))
                rolling_validation_prediction_ledger['Sparse_Poisson'].append(poisson_projected_value[0])
                
                tensor_time_series_representation = torch.FloatTensor(y_dimensional_training_vector).view(-1, 1, 1)
                pytorch_lstm_network = EndogenousRecurrentNetworkNode()
                gradient_optimizer_function = torch.optim.Adam(pytorch_lstm_network.parameters(), lr=0.005)
                huber_loss_function = nn.HuberLoss(delta=1.0)
                
                for _ in range(800):
                    gradient_optimizer_function.zero_grad()
                    network_output_state = pytorch_lstm_network(tensor_time_series_representation)
                    calculated_loss_metric = huber_loss_function(network_output_state, torch.FloatTensor(y_dimensional_training_vector).view(-1, 1))
                    calculated_loss_metric.backward()
                    gradient_optimizer_function.step()
                    
                with torch.no_grad():
                    current_tensor_view = tensor_time_series_representation[-1].view(1, 1, 1)
                    lstm_projected_value = pytorch_lstm_network(current_tensor_view).item()
                    rolling_validation_prediction_ledger['LSTM'].append(np.maximum(0, lstm_projected_value))

            sum_actual_ground_truths = np.sum(actual_validation_ground_truths)
            if sum_actual_ground_truths == 0:
                process_tracker_object.update(1)
                continue

            calculated_algorithm_errors = {}
            for algo_key, prediction_array in rolling_validation_prediction_ledger.items():
                sum_absolute_divergence = np.sum(np.abs(np.array(actual_validation_ground_truths) - np.array(prediction_array)))
                calculated_algorithm_errors[algo_key] = sum_absolute_divergence / sum_actual_ground_truths

            crowned_champion_identifier = min(calculated_algorithm_errors, key=calculated_algorithm_errors.get)
            
            final_extrapolated_vector = np.zeros(1)
            
            if crowned_champion_identifier in algorithm_architectures:
                algorithm_architectures[crowned_champion_identifier]._execute_tensor_ingestion_and_optimization(historical_subset_frame, constructed_exogenous_tensor)
                final_extrapolated_vector = algorithm_architectures[crowned_champion_identifier]._execute_extrapolation_projection(1, constructed_exogenous_tensor)
            else:
                x_dimensional_absolute_vector = np.arange(len(historical_subset_frame)).reshape(-1, 1)
                y_dimensional_absolute_vector = historical_subset_frame['q_val'].values
                x_dimensional_future_vector = np.array([[len(historical_subset_frame)]])
                
                scalar_absolute_transformation = StandardScaler()
                x_dimensional_absolute_scaled = scalar_absolute_transformation.fit_transform(x_dimensional_absolute_vector)
                x_dimensional_future_scaled = scalar_absolute_transformation.transform(x_dimensional_future_vector)
                
                if crowned_champion_identifier == 'Custom_Ensemble':
                    mlp_absolute_architecture = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=3000, early_stopping=True).fit(x_dimensional_absolute_scaled, y_dimensional_absolute_vector)
                    huber_absolute_architecture = HuberRegressor(max_iter=3000).fit(x_dimensional_absolute_scaled, y_dimensional_absolute_vector)
                    final_extrapolated_vector = np.maximum(0, (mlp_absolute_architecture.predict(x_dimensional_future_scaled) * 0.5) + (huber_absolute_architecture.predict(x_dimensional_future_scaled) * 0.5))
                elif crowned_champion_identifier == 'MLP':
                    mlp_absolute_architecture = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=3000, early_stopping=True).fit(x_dimensional_absolute_scaled, y_dimensional_absolute_vector)
                    final_extrapolated_vector = np.maximum(0, mlp_absolute_architecture.predict(x_dimensional_future_scaled))
                elif crowned_champion_identifier == 'Short_History_RF':
                    rf_absolute_architecture = RandomForestRegressor(n_estimators=200, random_state=42).fit(x_dimensional_absolute_scaled, y_dimensional_absolute_vector)
                    final_extrapolated_vector = np.maximum(0, rf_absolute_architecture.predict(x_dimensional_future_scaled))
                elif crowned_champion_identifier == 'Sparse_Poisson':
                    poisson_absolute_architecture = PoissonRegressor(max_iter=1000).fit(x_dimensional_absolute_scaled, y_dimensional_absolute_vector)
                    final_extrapolated_vector = np.maximum(0, poisson_absolute_architecture.predict(x_dimensional_future_scaled))
                elif crowned_champion_identifier == 'LSTM':
                    tensor_absolute_representation = torch.FloatTensor(y_dimensional_absolute_vector).view(-1, 1, 1)
                    absolute_lstm_network = EndogenousRecurrentNetworkNode()
                    absolute_gradient_optimizer = torch.optim.Adam(absolute_lstm_network.parameters(), lr=0.005)
                    absolute_loss_function = nn.HuberLoss(delta=1.0)
                    for _ in range(1500):
                        absolute_gradient_optimizer.zero_grad()
                        network_absolute_output = absolute_lstm_network(tensor_absolute_representation)
                        absolute_loss_metric = absolute_loss_function(network_absolute_output, torch.FloatTensor(y_dimensional_absolute_vector).view(-1, 1))
                        absolute_loss_metric.backward()
                        absolute_gradient_optimizer.step()
                    with torch.no_grad():
                        current_absolute_tensor = tensor_absolute_representation[-1].view(1, 1, 1)
                        final_extrapolated_vector = np.array([np.maximum(0, absolute_lstm_network(current_absolute_tensor).item())])

            serialized_dealer_dictionary = {
                'Dealer_Code': singular_dealer_id,
                'State': extracted_state_parameter,
                'Exclusive': extracted_exclusivity_parameter,
                'Winning_Algorithm': crowned_champion_identifier,
                'Rolling_Validation_WMAPE': f"{calculated_algorithm_errors[crowned_champion_identifier]:.2%}"
            }
            
            champion_prediction_array = rolling_validation_prediction_ledger[crowned_champion_identifier]
            for sequence_index, chronological_date in enumerate(self.backtest_validation_domain):
                actual_divergence_volume = actual_validation_ground_truths[sequence_index]
                predicted_divergence_volume = champion_prediction_array[sequence_index]
                calculative_divisor = actual_divergence_volume if actual_divergence_volume > 0 else (1.0 if predicted_divergence_volume > 0 else 1.0)
                monthly_error_ratio = np.abs(actual_divergence_volume - predicted_divergence_volume) / calculative_divisor
                serialized_dealer_dictionary[f"Val_WMAPE_{chronological_date.strftime('%b_%y')}"] = f"{monthly_error_ratio:.2%}"
            
            for index_p, chronological_date_p in enumerate(self.target_forecast_domain):
                serialized_dealer_dictionary[f"Forecast_{chronological_date_p.strftime('%b_%y')}"] = final_extrapolated_vector[index_p]
                
            compiled_output_dictionary_list.append(serialized_dealer_dictionary)
            process_tracker_object.update(1)

        process_tracker_object.close()

        final_dataframe_matrix = pd.DataFrame(compiled_output_dictionary_list)
        final_dataframe_matrix.to_excel("rolling_tournament_forecast_matrix.xlsx", index=False)

if __name__ == "__main__":
    orchestration_engine = MasterExecutionManifold(input_filepath_string='dealer_data.xlsx')
    orchestration_engine.execute_global_matrix_tournament()
