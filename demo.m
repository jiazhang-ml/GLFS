clear;clc; addpath(genpath('.\'))
    
    load Medical_data
    
    fi = find(train_target==-1); train_target(fi) = 0;

    optmParameter.alpha   = 0.2;  
    optmParameter.beta    = 1000;  
    optmParameter.gamma   = 10;  
    optmParameter.lamda   = 0.4;

%     optmParameter.rho     = 10;
    
    optmParameter.c  = 8;
    
    optmParameter.maxIter           = 200;
    optmParameter.minimumLossMargin = 0.001;
    
t0 = clock;    
[model_PRO,totalloss] = main_function(train_data,train_target',optmParameter);
time = etime(clock, t0);

t=50; %t=num_feature; 
%FS:
[dumb idx] = sort(sum(model_PRO.*model_PRO,2),'descend');
feature_idx = idx(1:t);

Num=10;Smooth=1;  
for i = 1:t
    fprintf('Running the program with the selected features - %d/%d \n',i,t);
    f=feature_idx(1:i);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,f),train_target,Num,Smooth);
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,macrof1,microf1,SubsetAccuracy,Outputs,Pre_Labels]=...
        MLKNN_test(train_data(:,f),train_target,test_data(:,f),test_target,Num,Prior,PriorN,Cond,CondN);
    HL_PRO(i)=HammingLoss;
    RL_PRO(i)=RankingLoss;
    OE_PRO(i)=OneError;
    CV_PRO(i)=Coverage;
    AP_PRO(i)=Average_Precision;
    MA_PRO(i)=macrof1;
    MI_PRO(i)=microf1;
    AC_PRO(i)=SubsetAccuracy;
end

save('PRO_Medical.mat','AP_PRO','RL_PRO','OE_PRO','AC_PRO','CV_PRO','HL_PRO','MA_PRO','MI_PRO','time');
