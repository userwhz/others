%% Calculate variance.

function res = main(x,f, indice, T)

addpath(genpath('./'))


% f = input('Please input file (E.g.: H2_4jw/H2_8jw/LiH_12jw/BeH2_14jw/H2O_14jw/NH3_16jw...)\n','s');

% f = strcat(f,'.txt');

%mole = input('Please input the molecule you want to use (BeH2 for ''BeH2'', H2OE for ''H2O'', and anything for others)','s');
mole = '';

%% write to the file
 tic
if x == 1
    fprintf('LBCS\n');
    [diagonal_var] = main_LBCS(f);
   % var = varianceLBCS();
    display(diagonal_var);
elseif x == 2
    fprintf('Grouping\n');
    LDFGroup(f, mole);
    var = varianceGroup();
    display(var);
elseif x == 3
    fprintf('LDF-OGM\n');
    [diag_var] = LDFGB(f, mole);
  %  var = varianceTotal();
    display(diag_var);
elseif x == 4
    fprintf('L1sampling\n');
    [var] = L1sampling(f, mole);
    display(var);
elseif x == 7
    fprintf('OGMV1 cutdown:\n');
    diagonal_var = main_CUTOGMV2(f,indice, T);
    display(diagonal_var);
end

timeElapsed = toc

res = timeElapsed;
end