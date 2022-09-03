function [model_PRO,totalloss] = main_function(X,Y,optmParameter)
%   Input
%       X               - a n by d data matrix, n is the number of instances and d is the number of features 
%       Y               - a n by l label matrix, n is the number of instances and l is the number of labels
%       optmParameter   - the optimization parameters for GLFS, a struct variable with several fields, 
%   Output
%
%       model_PRO  - a d by l Coefficient matrix

  
    % optimization parameters
    alpha            = optmParameter.alpha;
    lamda            = optmParameter.lamda;
    beta             = optmParameter.beta;
    gamma            = optmParameter.gamma;
%     rho              = optmParameter.rho;
    c                = optmParameter.c;

    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;
    
     [idx1,cc1] = kmeans(Y',5,'emptyaction','singleton');
     [idx2,cc2] = kmeans(X,c,'emptyaction','singleton');   
    for i = 1:c
        f = find(idx2 == i); 
        Yr = Y(f,:);
        for j = 1:length(f)
            temp(j,:) = mean(Yr);
        end
        CC{i} = temp;
        if length(f) >= 2
            Gr = 1 - pdist2( Yr, temp, 'cosine' );
            Rr = 1 - pdist2( Yr', Yr', 'cosine' );
            ind1 = find(isnan(Gr));Gr(ind1)=0;
            ind2 = find(isnan(Rr));Rr(ind2)=0;
            Grr{i} = Gr;
            Rrr{i} = Rr;
            clear f temp Xr Yr Gr Rr ind1 ind2;
        else
            Grr{i} = 1;
            Rrr{i} = 1;
            clear f temp Xr Yr;
        end
    end
       
    [num_ins num_dim] = size(X); num_label = size(Y,2);
    H = eye(num_ins) - 1 / num_ins * ones(num_ins, 1) * ones(num_ins, 1)';
    
    XTX = X'*H*X;
        
    M = rand(num_dim, num_label);    
    Z = Y;
    W_s = rand(num_dim, num_label); 
%     W_s   = (XTX + rho*eye(num_dim)) \ (X'*H*Z-XTX*M);
    W_s_1 = W_s;
    
    iter  = 1;
    oldloss = 0;
    bk = 1; bk_1 = 1; 
    Lip = sqrt(2*norm(XTX)^2 );
    while iter <= maxIter
       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       Gw_s_k = W_s_k - 1/Lip * (XTX*W_s_k + XTX*M - X'*H*Z);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;

       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,beta/Lip);
       ind = find(isinf(W_s)); W_s(ind) = 0;

       for i = 1:5
           f = find(idx1 == i);
           Zc = Z(:,f);
           W_sc = W_s(:,f);
           Mc = M(:,f);
           d = 0.5./sqrt(sum(Mc.*Mc, 2) + eps);
           D = diag(d);
           Mc =  (XTX + gamma*D+ eps*eye(num_dim))\(X'*H*Zc - XTX*W_sc);
           for j = 1:length(f)
               M(:,f(j)) = Mc(:,j);
           end
           clear f Zr W_sc Mc;
       end
       localloss = 0;
       for i = 1:c
           f = find(idx2 == i); 
           for t = 1:length(f)
               U(t,t) = Grr{i}(t,1);
           end
           Xr = X(f,:); Yr = Y(f,:);
           if length(f) >= 2
               Hr=H(1:length(f),1:length(f));
               A = Hr + alpha*U;
               B = ((1-lamda)*eye(num_label)+lamda*Rrr{i})*((1-lamda)*eye(num_label)+lamda*Rrr{i})';
               C = -(Hr*Xr*(W_s+M) + alpha*U*CC{i} + Yr*((1-lamda)*eye(num_label)+lamda*Rrr{i}'));
               indA = find(isnan(A));A(indA)=0;
               indB = find(isnan(B));B(indB)=0;
               indC = find(isnan(C));C(indC)=0;
               Zr = lyap(A, B, C);
               localloss = localloss + alpha*trace((Zr-CC{i})'*U*(Zr-CC{i})) + (norm((Zr*((1-lamda)*eye(num_label)+lamda*Rrr{i}) - Yr), 'fro'))^2;
               for j = 1:length(f)
                   Z(f(j),:) = Zr(j,:);
               end
               clear f U Xr Yr Gr Rr ind1 ind2 Hr A B C Zr indA indB indC;
           else
               Zr = Yr;
               Z(f(1),:) = Zr;
               localloss = localloss + alpha*trace((Zr-CC{i})'*U*(Zr-CC{i})) + (norm((Zr*((1-lamda)*eye(num_label)+lamda*Rrr{i}) - Yr), 'fro'))^2;
               clear f U Xr Yr Gr Rr Zr;
           end
       end
       
       specificloss    = trace((X*W_s + X*M - Z)'*H*(X*W_s + X*M - Z));
       sparesW_s       = sum(sum(W_s~=0));
       sparesM         = sum(sqrt(sum(M.*M,2)+eps));
       totalloss(iter) = specificloss + localloss + beta*sparesW_s + gamma*sparesM; 
       
       if abs((oldloss - totalloss(iter))/oldloss) <= miniLossMargin
           break;
       elseif totalloss(iter) <= 0
           break;
       else
           oldloss = totalloss(iter);
       end
       iter = iter + 1;
    end
    
    model_PRO = W_s + M;