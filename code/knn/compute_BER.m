function [ BER, err ] = compute_BER(y_true, y_hat, C )
%COMPUTE_BER Compute Balance Error Rate
% Nc     - number of examples in each class
% y_true - true values
% y_hat  - predicted values

    
    % parameters
    N = 1:C;
    for c = 1:C
        N(c) = length( find(y_true==c ));
    end
    
    % compute error
    BER = 0;
    err = zeros(C, 1);
    for c = 1:C
        % calculate error for each class
        idx = find(y_true == c);
        err(c) = length(find(y_true(idx) ~= y_hat(idx)))/N(c);
        % add contribution to BER
        BER = BER + err(c);
    end
    BER = BER/C;

end

