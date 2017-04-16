clear;
filename = 'mss_transects_test_data.csv';
test_data = importdata(filename);

data = zscore(test_data.data(:,2:5));

% Use unnormalized data to calculate average percentage error
fish_rich_raw = test_data.data(:,5);
rich_mean = mean(fish_rich_raw);
rich_std = std(fish_rich_raw);

% Break down data by columns for reassembling
intercept = ones(5,1);
coral_prop = data(:,1);
ldim_abund = data(:,2);
fish_abund = data(:,3);
fish_rich = data(:,4);

rich_data_points = ...
    [   intercept, ...
        coral_prop, ...
        ldim_abund, ...
        fish_rich];
[rich_rows, rich_cols] = size(rich_data_points);

% Fish species richness
% Experiment 1: Ridge Regression
for a = 1
if 1
    cv_fold_k = 5;
    X = [intercept,coral_prop,ldim_abund]';
    Y = fish_rich;
    Lambda = logspace(-1.5,0,15);
    CVMdl_ridge = fitrlinear(...
                        X,Y,...
                        'ObservationsIn','columns',...
                        'KFold',cv_fold_k,...
                        'Lambda',Lambda,...
                        'Learner','leastsquares',...
                        'Solver','bfgs',...
                        'Regularization','ridge');
    mse_ridge = kfoldLoss(CVMdl_ridge);
end
end
% Experiment 2: lasso regression
for a = 1
if 1
    cv_fold_k = 5;
    X = [intercept,coral_prop,ldim_abund]';
    Y = fish_rich;
    Lambda = logspace(-1.5,0,15);
    % Train a cross-validated linear model
    CVMdl_lasso = fitrlinear( X,Y,...
                        'ObservationsIn','columns',...
                        'KFold',cv_fold_k,...
                        'Lambda',Lambda,...
                        'Learner','leastsquares',...
                        'Solver','sparsa',...
                        'Regularization','lasso');
    % Calculate the mean square error
    mse_lasso = kfoldLoss(CVMdl_lasso);
    
    % Create a double-figure
    figure
    plot(log10(Lambda),log10(mse_ridge),'-o',log10(Lambda),log10(mse_lasso),'-o','LineWidth',2);
    title('Mean Square Error vs. Regularization Strength')
    ylabel('log_{10} MSE')
    xlabel('log_{10} Lambda')
    legend('Ridge Regression','LASSO Regression','Location','NorthWest');
end
end

% The regression has been selected: LASSO.
for a = 1
if 1
    cv_fold_k = 5;
    [mse_min,I] = min(mse_lasso);
    X = [intercept,coral_prop,ldim_abund]';
    Y = fish_rich;
    CVMdl_lasso = fitrlinear(...
                        X,Y,...
                        'ObservationsIn','columns',...
                        'KFold',cv_fold_k,...
                        'Lambda',Lambda(I),...
                        'Learner','leastsquares',...
                        'Solver','sparsa',...
                        'Regularization','lasso');
    mse = kfoldLoss(CVMdl_lasso);
    rmse = sqrt(mse);
    
    Y_hat = kfoldPredict(CVMdl_lasso);
    error = Y_hat - Y;
    error_real = (error * rich_std) + rich_mean;
    abs_err_perc = abs(error_real ./ fish_rich_raw);
    % Median absolute percentage (MAPE)
    mape = 100 * median(abs_err_perc);
end
end
