function A = afun(W,X_tr,phi,sigma2)
    A = X_tr'*W;
    %tmpmat = bsxfun(@times,X_tr,phi'); %DxN
    A = phi.*A;
    %A = tmpmat*A;
    A = X_tr*A;
    %A = A + diag(ones(D,1)./sigma2)*W;
    A = A + W./sigma2;
end