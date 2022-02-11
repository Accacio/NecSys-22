function [Phi, pi_new] = update_parameters(Theta, Lambda, Responsabilities)
% UPDATE_PARAMETERS -
    modes=size(Responsabilities,1);
    [c O]=size(Theta);

    Upsilon=kron(ones(c,1),eye(c));
    Delta=kron(eye(O),ones(1,c));
    G=kron(ones(1,O),eye(c));
    Y=kron(G,ones(c,1));
    Omega=sparse([(Upsilon*Theta*Delta).*Y; G]');

    pi_new(:)=sum(Responsabilities,2)/O;
    for i=1:modes
        responsabilities=Responsabilities(i,:);
        resp2=cellfun(@(x) x*eye(c),mat2cell(responsabilities',ones(1,O)),'UniformOutput',0);
        Gamma=sqrt(sparse(blkdiag(resp2{:})));

        Phi(i,:)=-((Gamma*Omega)\(Gamma*Lambda(:)));
    end

end
