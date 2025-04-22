% Voice Biometric System in MATLAB

% Add necessary paths (you need the Voicebox toolbox for MFCC)
addpath('voicebox'); % Make sure you have Voicebox toolbox for MATLAB

% Parameters
numComponents = 16; % Number of Gaussian components
numMFCC = 12; % Number of MFCC features
fs = 16000; % Sampling frequency

% Load training data (enrollment phase)
trainingData = loadAudioData('data/training', fs);

% Train GMM model
gmmModel = trainGMM(trainingData, numComponents, numMFCC);

% Load testing data (authentication phase)
testingData = loadAudioData('data/testing', fs);

% Authenticate using GMM model
authenticate(gmmModel, testingData, numMFCC);

% Function to load audio data
function data = loadAudioData(directory, fs)
    audioFiles = dir(fullfile(directory, '*.wav'));
    data = cell(1, length(audioFiles));
    for i = 1:length(audioFiles)
        [audio, originalFs] = audioread(fullfile(directory, audioFiles(i).name));
        if originalFs ~= fs
            audio = resample(audio, fs, originalFs); % Resample to target frequency
        end
        data{i} = audio;
    end
end

% Function to train GMM model
function gmmModel = trainGMM(trainingData, numComponents, numMFCC)
    allMFCC = [];
    for i = 1:length(trainingData)
        mfccFeatures = extractMFCC(trainingData{i}, numMFCC);
        allMFCC = [allMFCC; mfccFeatures];
    end
    gmmModel = fitgmdist(allMFCC, numComponents, 'CovarianceType', 'diagonal', 'RegularizationValue', 0.01);
end

% Function to authenticate using GMM model
function authenticate(gmmModel, testingData, numMFCC)
    for i = 1:length(testingData)
        mfccFeatures = extractMFCC(testingData{i}, numMFCC);
        logLikelihood = sum(log(pdf(gmmModel, mfccFeatures)));
        fprintf('Log-Likelihood for test sample %d: %f\n', i, logLikelihood);
    end
end

% Function to extract MFCC features
function mfccFeatures = extractMFCC(audio, numMFCC)
    % Apply pre-emphasis filter
    preEmphasized = filter([1 -0.97], 1, audio);
    
    % Compute MFCC features
    mfccFeatures = melcepst(preEmphasized, 16000, 'M', numMFCC, floor(3*log(16000)), 0.025*16000, 0.01*16000);
end
