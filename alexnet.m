clear all;
close all;

data = load('mnist_train.csv');
train_labels = data(1:2000,1);
y = zeros(10, 2000);
for i = 1:2000
    y(train_labels(i)+1,i) = 1;
end
train_images = data(1:2000,2:785); 
train_images = train_images/255;
train_images = train_images';
Y = reshape(train_images, 28, 28, 1, 2000);
Y = imresize(Y,[227 227]);
Y = repmat(Y,[1 1 3]);

test = load('mnist_test.csv');
test_labels = test(1:1000,1);
x = zeros(10,1000);
for i = 1:1000
    x(test_labels(i)+1,i) = 1;
end
test_images = test(1:1000,2:785); 
test_images = test_images/255;
test_images = test_images';
X = reshape(test_images, 28, 28, 1, 1000);
X = imresize(X,[227 227]);
X = repmat(X,[1 1 3]);



options = trainingOptions('sgdm', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.9, ...
    'LearnRateDropPeriod',2, ...
    'Shuffle','once', ...
    'Verbose',true, ...
    'ValidationFrequency',30,'ValidationPatience', Inf, 'Plots','training-progress'); 
layers = [
    imageInputLayer([227 227 3],"Name","data")
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","norm1","K",1)
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    reluLayer("Name","relu2")
    crossChannelNormalizationLayer(5,"Name","norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu3")
    groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu4")
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu5")
    maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2)
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];
plot(layerGraph(layers));

netTransfer=trainNetwork(Y , categorical(train_labels), layers, options);
[Ypred,score] = classify(netTransfer, Y);
accuarry = mean(Ypred == categorical(test_labels)')


