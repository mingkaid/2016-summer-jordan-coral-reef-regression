clear;
filename = 'Wagner_et_al_test_data.csv';
test_data = importdata(filename);

% normalize data
data = zscore(test_data.data);
[data_rows, data_cols] = size(data);

% extract columns for recombination
intercept = ones(data_rows,1);
patch_vol = data(:,1);
cave_prop = data(:,2);
coral_prop = data(:,3);
Ldim_abundance = data(:,4);
fish_abundance = data(:,5);
fish_richness = data(:,6);

% assemble dataset
data_points = ...
        [intercept, ...
        cave_prop, ...
        coral_prop,  ...
        coral_prop.*cave_prop, ...
        fish_abundance]; 
[rows,cols] = size(data_points);

% A naive coefficient vector to start the regression
coeff_prior = ones(1,cols-1);
coeff_prior = coeff_prior/100;

% two lambdas refer to strength of Ridge and LASSO 
% regularization terms, respectively
best_l = [0,0];
best_abund_model_coeff = zeros(1,cols-1);
best_test_mse = Inf;
best_test_error_percent = Inf;
lambda_1 = 0:1e-2:0.2;
lambda_2 = 0:1e-2:0.2;

split_parts = 5;
indices = crossvalind('Kfold', rows, split_parts);
for l_1 = lambda_1
    for l_2 = lambda_2
        
        avg_test_mse = 0;
        avg_model = zeros(1,cols-1);
        avg_test_error_percent = 0;
        % N-fold cross validation
        for i = 1:split_parts
            % Create the test and training sets
            test = (indices == i); 
            train = ~test;
            
            test_set = zeros(sum(test), cols);
            test_set_index = 1;
            train_set = zeros(sum(train), cols);
            train_set_index = 1;
            
            % Build the test and training sets
            for data_index = 1:rows
                if test(data_index)
                    test_set(test_set_index, :) = data_points(data_index, :);
                    test_set_index = test_set_index + 1;
                end
                
                if train(data_index)
                    train_set(train_set_index, :) = data_points(data_index, :);
                    train_set_index = train_set_index + 1;
                end
            end
            
            % Minimize the total error function to find the model coefficients
            % Using Ridge and LASSO regularization
            % Train the model
            train_features = train_set(:, 1:cols-1);
            train_response = train_set(:, cols);
            abundance_total_error_fun = ...
                @(coeff)sum((coeff*train_features')' - ... 
                train_response).^2 / rows + ...
                l_2*norm(coeff)*norm(coeff) + ...
                l_1*norm(coeff);
            % Minimize error function
            model_coeff = ...
                fmincon(abundance_total_error_fun, ...
                        coeff_prior, ...
                        [],[], ...
                        [],[],... 
                        zeros(1,cols-1),Inf(1,cols-1));
            
            % Test the model
            test_features = test_set(:, 1:cols-1);
            test_response = test_set(:, cols);
            test_mse = sum(((model_coeff*test_features')' -...
                              test_response).^2) / rows;
            avg_test_mse = avg_test_mse + test_mse / split_parts;
            avg_model = avg_model + model_coeff / split_parts;
            
            test_error = (model_coeff*test_features')' - test_response;
            test_error_percent = ...
                                mean(((model_coeff*test_features')' -...
                                test_response) ./ test_response);
            avg_test_error_percent = avg_test_error_percent + test_error_percent / split_parts;
        end
        
        % find the combination of parameters that results in the lowest 
        % mean-squared-error
        if avg_test_mse < best_test_mse
            best_test_error_percent = avg_test_error_percent;
            best_test_mse = avg_test_mse;
            best_abund_model_coeff = avg_model;
            best_l = [l_1, l_2];
        end
    end
end