function [f, df] = gradientB(B1, constant, Z, lambda, rho1, rho2, rho3, rho4)

[n, bit] = size(constant);
B = reshape(B1, [n, bit]);
one_vector = ones(n, 1);
temp = one_vector' * B;
temp1 = B' * B - n * eye(bit, bit);
% lambda = diag(lamda .^ -1);
f = (rho1 + rho2 + 2) / 2 * trace(B' * B) - trace(B' * Z * lambda * (Z' * B)) ...
    + trace(constant' * B) + rho3 / 2 * trace(temp' * temp) ...
    + rho4 / 4 * trace(temp1' * temp1);

if nargout > 1
    g = 2 * (B - Z * lambda * (Z' * B)) + constant + (rho1 + rho2) * B ...
        + rho3 * one_vector * temp + rho4 * B * temp1;
    df = g(:);
end
