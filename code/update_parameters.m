function [Phi, pi_new] = update_parameters(Theta, Lambda, Responsabilities)
% UPDATE_PARAMETERS -
    modes=size(Responsabilities,1);
    [n N]=size(Theta);

    Upsilon=kron(ones(n,1),eye(n));
    Delta=kron(eye(N),ones(1,n));
    G=kron(ones(1,N),eye(n));
    Y=kron(G,ones(n,1));
    Omega=sparse([(Upsilon*Theta*Delta).*Y; G]');

    N_k=sum(Responsabilities,2);
    pi_new(:)=N_k/N;
    for i=1:modes
        responsabilities=Responsabilities(i,:);
        resp2=cellfun(@(x) x*eye(n),mat2cell(responsabilities',ones(1,N)),'UniformOutput',0);
        Gamma=sqrt(sparse(blkdiag(resp2{:})));

        Phi(i,:)=-((Gamma*Omega)\(Gamma*Lambda(:)));
    end

end
