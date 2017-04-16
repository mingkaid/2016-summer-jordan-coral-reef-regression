clear;
filename = 'mss_transects_test_data.csv';
test_data = importdata(filename);

% normalize data
data = zscore(test_data.data(:,2:5));

% Use unnormalized data to calculate average percentage error
fish_abundance_raw = test_data.data(:,4);
abund_mean = mean(fish_abundance_raw);
abund_std = std(fish_abundance_raw);

% Break down data by columns for reassembling
intercept = ones(5,1);
coral_prop = data(:,1);
ldim_abund = data(:,2);
fish_abund = data(:,3);
fish_rich = data(:,4);

abund_data_points = ...
    [   intercept, ...
        coral_prop, ...
        fish_abund];
[abund_rows, abund_cols] = size(abund_data_points);

% Fish abundance
% Experiment 1: Ridge Regression
for a = 1
if 1
    cv_fold_k = 5;
    X = [intercept,coral_prop]';
    Y = fish_abund;
    Lambda = logspace(-3,0,15);
    % Train a cross-validated linear model
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
    X = [intercept,coral_prop]';
    Y = fish_abund;
    Lambda = logspace(-3,-0,15);
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
    % Visualize evolvement of MSE for Ridge and lasso
    figure
    plot(log10(Lambda),log10(mse_ridge),'-o',log10(Lambda),log10(mse_lasso),'-o','LineWidth',2);
    title('Mean Square Error vs. Regularization Strength')
    ylabel('log_{10} MSE')
    xlabel('log_{10} Lambda')
    legend('Ridge Regression','LASSO Regression','Location','NorthWest');
end
end

% The regression has been selected: Ridge with log(-1) regularization term.
for a = 1
if 1
    cv_fold_k = 5;
    X = [intercept,coral_prop]';
    Y = fish_abund;
    CVMdl_ridge = fitrlinear(...
                        X,Y,...
                        'ObservationsIn','columns',...
                        'KFold',cv_fold_k,...
                        'Lambda',0.1,...
                        'Learner','leastsquares',...
                        'Solver','bfgs',...
                        'Regularization','ridge');
    mse = kfoldLoss(CVMdl_ridge);
    rmse = sqrt(mse);
    
    Y_hat = kfoldPredict(CVMdl_ridge);
    error = Y_hat - Y;
    error_real = (error * abund_std) + abund_mean;
    abs_err_perc = abs(error_real ./ fish_abundance_raw);
    % Median absolute percentage (MAPE)
    mape = 100 * median(abs_err_perc);
    
    % Below is to visualize a predicted line against a proposed 
    % model that is closer to logistic regression
    Mdl_ridge = fitrlinear(...
                        X,Y,...
                        'ObservationsIn','columns',...
                        'Lambda',0.1,...
                        'Learner','leastsquares',...
                        'Solver','bfgs',...
                        'Regularization','ridge');
    X_line = linspace(-1.5,1.5)';                
    X_predict = [ones(100,1),X_line];
    Y_line = predict(Mdl_ridge,X_predict);
    
    % Make a sigmoid plot
    sigfunc = @(A, x)(A(1) ./ (1 + exp(-A(2)*x)));
    A0 = [1,1]; %// Initial values fed into the iterative algorithm
    A_fit = nlinfit(X_line, Y_line, sigfunc, A0);
    set_middle = (max(Y_line)+min(Y_line)) / 2;
    Y_sig = sigfunc(A_fit,X_line)*3.75 - 1.2;
    
    figure
    scatter(coral_prop,fish_abund,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5);
    hold on
    plot(X_line,Y_line,X_line,Y_sig,'LineWidth',2);
    title('Fish Abundance vs. Live Coral Cover')
    ylabel('Fish Abundance z-Score')
    xlabel('Live Coral Cover z-Score')
    legend('Observation','Ridge-Regression-Fitted Line','Possible Sigmoid Function',...
           'Location','NorthWest');
    hold off
end
end
