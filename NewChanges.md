
# Document for Changes to Informer Model

This document outlines the modifications made to the *Dataset_Custom* class, *Exp_Informer* class and perhaps a few other classes to enhance their functionality, adaptability and customization, for different use cases. The changes include adjustments to parameter defaults, introduction of new parameters, enabling detailed tracking of performance metrics and improvements in the data scaling process.

## Changes to Dataset_Custom Class


### Parameter Changes:
        
        features: Default value changed from 'S' to 'MS', enabling multivariate scaling by default.
        data_path: Default value changed from 'ETTh1.csv' to 'data.csv' to reflect a more generic placeholder for data files.
        target: Default value changed from 'OT' to 'Close', typically used in financial datasets to denote closing prices.
        freq: Frequency of data aggregation changed from hourly ('h') to business days ('b'), which is common in financial time series analysis.

### New Parameters Introduced:

        kind_of_scaler: Allows specifying the type of scaler to use ('Standard' or 'MinMax'). If none provided, defaults to 'Standard'.
        test_size: Allows specifying the proportion of the dataset to be reserved for testing. The remaining data minus the test set is used for training.

### Size Configuration:

        Updated default sequence lengths to more compact settings, presumably to fit smaller or more specific windows of time series data:
            seq_len: From 24*4*4 to 1*5*2 (presumed to represent a smaller window for analysis).
            label_len and pred_len: Both simplified from 24*4 to 1*1, indicating shorter periods for labels and predictions.

### Data Handling Enhancements:
        
        Dynamic Train/Test Split: The training dataset size is dynamically adjusted based on the test_size, with the validation set size being calculated as the residual.
        Enhanced Scaling Logic: Individual columns can now be scaled using either StandardScaler or MinMaxScaler. The scaler for the target column is preserved separately, allowing for easy inverse transformations if needed.

### Code Structure and Execution:
        
        __read_data__ Method: This method has been restructured to accommodate the flexible scaling of individual columns based on the kind_of_scaler setting.
        Scaler Persistence: The scaler used for the target column is saved as a .pkl file, facilitating model deployment or further analysis without re-training the scaler.

### Enhanced Functionality

    Flexibility in Data Scaling: Users can now choose how each column is scaled, which is particularly useful in datasets where different features have different ranges and distributions.
    Improved Customizability: With parameters like kind_of_scaler and test_size, users have more control over the preprocessing and partitioning of the data, allowing the class to be better tailored to specific analytical needs.
    Support for Diverse Data Frequencies: By supporting different frequencies like business days ('b'), the class becomes more suitable for financial time series analysis, where non-trading days do not need to be included.


## Changes to Exp_Informer Class


### Constructor ```__init__``` Enhancements:
        Tracking Metrics: New lists train_losses, test_losses, actual_test_values, and predicted_test_values have been introduced to track detailed training and testing metrics throughout the model's operation.

### Data Loader Configuration:
        Shuffling Flags: The approach to handling data shuffling is now more customizable with shuffle_for_test, shuffle_for_pred, and shuffle_for_train flags introduced. These flags are pulled from the args passed during initialization, allowing for dynamic adjustments based on experimental needs.
        Batch and Frequency Handling: Enhanced flexibility in configuring batch sizes and frequency parameters for different operation modes (test, pred, and default training mode).

### Optimizer Selection:
        Extended Optimizer Options: The method _select_optimizer has been significantly expanded to include a wide range of optimizers. This allows for a more fine-tuned approach to optimization based on specific use cases or performance objectives.
        Customization Through args: The optimizer to be used can now be specified through args.kind_of_optim, enabling the selection of optimizers like AdamW, SparseAdam, SGD, RMSprop, RAdam, NAdam, LBFGS, Adamax, ASGD, Adadelta, and Adagrad. The default optimizer remains Adam if none is specified.


## Introduction of Custom and Standard Loss Functions:
        
### Custom Loss Functions: 
        New loss functions such as WeightedMeanAbsolutePercentageError, SymmetricMeanAbsolutePercentageError, RMSELoss, QuantileLoss, HuberLoss, and PinballLoss have been introduced. These functions are sourced from utils.criter, a module presumably containing implementations tailored to specific use cases, especially for handling regression tasks where such metrics are crucial.
        Standard Loss Functions: Integration of PyTorch's native loss functions like nn.L1Loss (Mean Absolute Error) and nn.MSELoss (Mean Squared Error) ensures that users have access to well-established methods for baseline comparisons.

### *Dynamic Loss Function Selection*
        The method leverages self.args.criter, a parameter from the model's argument list, to dynamically select the appropriate loss function based on user input. This design ensures flexibility and easy customization of the model training process.

By parameterizing the choice of loss function, the _select_criterion method allows users to easily experiment with different types of errors and loss metrics, which is particularly useful in exploring the best setups for various prediction tasks.


## Conclusion 
These enhancements make the Exp_Informer class a more robust and versatile tool for handling complex machine learning workflows. By enabling detailed tracking of performance metrics and offering customizable data handling and optimization strategies, the class is well-suited to meet the diverse needs of users engaged in advanced data analysis and modeling tasks. The introduction of dynamic shuffling flags and a wide selection of optimizers provides flexibility and fine-tuning capabilities essential for optimizing model training and evaluation processes.


The enhanced _select_criterion method provides the Exp_Informer class with a robust and flexible framework for selecting loss functions, critical for optimizing model performance across different types of prediction tasks. This update facilitates more refined model tuning and enables users to achieve better alignment with specific performance metrics and objectives. By supporting a wide range of loss functions, the class becomes significantly more versatile and powerful, suitable for advanced machine learning challenges.