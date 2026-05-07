import numpy as np
from tqdm.asyncio import tqdm

class ConformalForecastingEvaluator:
    def __init__(self, predicted_links, actual_links, non_exist_links, miscoverage_level, method = "classic", dataset_name=None, model_name=None):
        self.predicted_links = predicted_links
        self.actual_links = actual_links
        self.non_exist_links = non_exist_links
        self.miscoverage_level = miscoverage_level
        self.method = method
        self.dataset_name, self.model_name = dataset_name, model_name
    
    def get_threshold(self, lambda_decay=0.001):
        if self.method == "classic":
            return self.get_threshold_from_scores(self.miscoverage_level)
        if self.method == "adaptive":
            return  None # no fixed threshold for adaptive method
        elif self.method == "weighted":
            return self.get_threshold_from_weighted_scores(self.miscoverage_level, lambda_decay)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
    def get_threshold_from_scores(self, miscoverage_level):
        errors = []
        for src, dst, score, ts in self.predicted_links:
                errors.append(1 - score)

        # take the miscoverage_level quantile of the error on the calibration set as threshold
        n_cal = len(errors)
        if int((1 - miscoverage_level) * (n_cal + 1)) >= n_cal:
            threshold = np.max(errors) + 1e-6  # set threshold above max error to ensure miscoverage <= desired level
        else:
            threshold = np.quantile(errors, int((1 - miscoverage_level) * (n_cal + 1)) / n_cal) 
        return threshold

    def get_threshold_from_weighted_scores(self, miscoverage_level, lambda_decay=0.001):
        weights = self.compute_weights(lambda_decay=lambda_decay)

        errors = []
        for src, dst, score, ts in self.predicted_links:
                errors.append(1 - score)
        
        n_cal = len(errors)
        normalized_weights = weights / np.sum(weights)
        # empirical weighted quantile computation
        sorted_indices = np.argsort(errors)
        sorted_errors = np.array(errors)[sorted_indices]
        sorted_weights = normalized_weights[sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights)
        threshold_index = np.searchsorted(cumulative_weights, int((1 - miscoverage_level) * (n_cal + 1)) / n_cal)
        threshold = sorted_errors[threshold_index]
        return threshold

    def compute_weights(self, lambda_decay=0.001):
        ts = np.array([ts for _, _, _, ts in self.predicted_links])
        t_max = np.max(ts)
        weights = np.exp(-lambda_decay * (t_max - ts)) 
        return weights

    def evaluate(self, test_predicted_links, test_actual_links, test_non_exist_links, lr=0.005):
        # Get the threshold
        threshold = self.get_threshold()

        if self.method == "adaptive":
            return self.evaluate_adaptive(test_predicted_links, test_actual_links, test_non_exist_links, lr)
        elif self.method == "weighted" or self.method == "classic":
            return self.evaluate_classic(test_predicted_links, test_actual_links, test_non_exist_links, threshold)
        else:
            raise ValueError(f"Unknown method: {self.method}")  
        
    def evaluate_classic(self, test_predicted_links, test_actual_links, test_non_exist_links, threshold):
        y_score = []
        for src, dst, score, ts in test_predicted_links:
            y_score.append(score)
        errors = 1 - np.array(y_score, dtype=np.float32)
        miscoverage_mask = errors > threshold
        miscoverage = np.mean(miscoverage_mask)
        return miscoverage, miscoverage_mask, self.miscoverage_level
        

    def evaluate_adaptive(self, test_predicted_links, test_actual_links, test_non_exist_links, lr=0.005, batch_size=256):
        miscoverage_level_list = [self.miscoverage_level]
        threshold_list = [self.get_threshold_from_scores(miscoverage_level_list[-1])]
        miscoverage_mask = []
        for idx, (src, dst, score, ts) in enumerate(tqdm(test_predicted_links)):
            error = 1 - score
            miscoverage_mask.append(error > threshold_list[-1])
           
            if (idx + 1) % batch_size == 0:
                # update miscoverage level and threshold after each batch
                batch_miscoverage = np.mean(miscoverage_mask[-batch_size:])
                if batch_miscoverage > self.miscoverage_level:
                    new_miscoverage_level = miscoverage_level_list[-1] + lr * (miscoverage_level_list[-1] - batch_miscoverage)
                else:
                    new_miscoverage_level = miscoverage_level_list[-1] + lr * miscoverage_level_list[-1]
                miscoverage_level_list.append(new_miscoverage_level)
                threshold_list.append(self.get_threshold_from_scores(new_miscoverage_level))
        miscoverage = np.mean(miscoverage_mask)
        return miscoverage, miscoverage_mask, miscoverage_level_list
    
