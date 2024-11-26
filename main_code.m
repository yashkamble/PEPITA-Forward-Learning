% Parameters
clear all;
clc;
rng(7);
inputSize = 64; % Number of input neurons (64 for 8x8 images)
hiddenLayerSize = 50; % Number of neurons in the hidden layer
outputSize = 10; % Number of output neurons (digits 0-9)
batchSize = 100; % Batch size (adjust as needed)
numBatches = 600; % Total number of batches
eta = 0.5; % Learning rate
% perturbation = 0.0001; % Perturbation for gradient estimation
numEpochs = 3; % Number of epochs for training (adjust as needed)


% Load the data
load('MNIST_TrainSet_0to1_8x8pixel.mat'); % Load image data
load('MNIST_TrainSet_Label.mat'); % Load labels

Images = number'; % Transpose to get images of size (60000 x 64)
shuffledIndices = randperm(60000);
allImages = Images(shuffledIndices, :); % Shuffle data
labelt = label';
labelt = labelt(shuffledIndices, :);
allLabels = labelt' + 1; % Convert labels to 1-based indexing if needed

% Initialize weights and biases
weightsInputHidden = randn(inputSize, hiddenLayerSize) ;
weightsHiddenOutput = randn(hiddenLayerSize, outputSize);

W{1} = weightsInputHidden;
W{2} = weightsHiddenOutput;

sd = sqrt(6 / 64);
B = (rand(64, 10) * 2 * sd - sd) * 0.05;

% Accuracy storage
accuracy = zeros(1, numEpochs * numBatches);

for epoch = 1:numEpochs
    for batch = 1:numBatches
        fprintf('Epoch %d, Processing batch %d...\n', epoch, batch);
        % Accuracy code goes here
        correct =0;
        for img = 1:60000
            curr_image = number(:,img)';
            out_L1=[curr_image]*W{1};
            act_L1=1./(1+exp(-out_L1));
            out_L2=[act_L1]*W{2};
            f_prudvi=1./(1+exp(-out_L2));
            [M, I] = max(f_prudvi);
            %disp(I);
            if((I-1) == label(img))
                correct = correct + 1;
            end
        end
        accuracy(1,(epoch-1)*numBatches + batch) = 100*correct/60000;
        disp(accuracy(1,(epoch-1)*numBatches + batch))
        startIdx = (batch*batchSize - batchSize);
        
        w1_temp =  zeros(inputSize, hiddenLayerSize);
        w2_temp =  zeros(hiddenLayerSize, outputSize);
        w_temp{1} = w1_temp;
        w_temp{2} = w2_temp;
        for imag = 1 : batchSize
            startIdx_image = startIdx + imag;
            current_image = allImages(startIdx_image,:);
            current_label = allLabels(startIdx_image);
            trueLabel =  zeros(1, 10);  % Create target vector     
            trueLabel(current_label) = 1;   
            delta_w_singular = forward_modulated_pass(current_image, trueLabel, W, 2, eta, B);
            w_temp{1} = w_temp{1} + delta_w_singular{1};
            w_temp{2} = w_temp{2} + delta_w_singular{2};
        end
        w_temp{1} = w_temp{1}/batchSize;
        w_temp{2} = w_temp{2}/batchSize;
        W = update_weights(W, w_temp, eta);
    end
end
figure
% Plot training accuracy
plot(accuracy, '-');
axis square;
xlabel('Batches');
ylabel('Training Accuracy (%)');

% --- Helper Functions ---

function delta_W = forward_modulated_pass(x, y, W, L, eta, F)
    % Standard forward pass
    h{1} = x; % Initialize input layer
    for l = 2:L+1
        h{l} = sigmoid(h{l-1} * W{l-1}); % Forward propagation
    end
    
    e = h{L+1} - y; % Compute error
    
    % Modulated forward pass
    h_err{1} = x - (F * e')'; % Modulated input
    h_err{2} = sigmoid(h_err{1} * W{1});
    h_err{3} = sigmoid(h_err{2} * W{2});
    
    % Weight updates for SGD
    delta_W{1} = (x - (F * e')')' * (h{2} - h_err{2});
    delta_W{2} = h_err{2}' * e;
  
end

function out = sigmoid(x)
    out = 1 ./ (1 + exp(-x));
end

% Function: Update Weights
function W = update_weights(W, delta_W, eta)
    
    %noise_W1 = normrnd(0, 0.1, size(W{1}));
    %noise_W2 = normrnd(0, 0.1, size(W{2}));
    noise_W1 = 0;
    noise_W2 = 0;

    % Apply gradient updates with noise
    W{1} = W{1} - eta * delta_W{1} + noise_W1;
    W{2} = W{2} - eta * delta_W{2} + noise_W2;
end
