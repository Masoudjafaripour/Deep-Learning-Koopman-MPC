% Written by Masoud Jafaripour - mjafaripour110@gmail.com
% lifting section adopted and modied -- originally by Milan Korda and Igor Mezic in  https://github.com/MilanKorda/KoopmanMPC
%
%% RC Circut
clc; clear all; close all;
rng(2141444) % Masoud: for randomness?

tic
%% *************************** Dynamics ***********************************
a11 = 2.26e-3;
a12 = 1.99e-3;
a21 = 3.62e-5;
a22 = 3.62e-5;

b1 = 1.20e-5;

c11 = 1.20e-5;
c12 = 1.20e-5;
c13 = 1.20e-5;
c21 = 2.69e-4;
c22 = 0;

% nonlinear term (bilinear) -->> x(1,:) - 0.05
f_u =  @(t,x,u)([ -a11*x(1,:).^2 + a12*x(2,:) + b1*(u(4,:).*1) + c11*u(1,:) + c12*u(2,:) + c13*u(3,:); ...
    a21*x(1,:) - a22*x(2,:) + c21*u(1,:) + c22*u(2,:)] );
n = 2;
m = 4; % number of control inputs - in case A -->> Qdot = u4


%% ************************** Discretization ******************************

deltaT = 0.01;
%Runge-Kutta 4
k1 = @(t,x,u) ( f_u(t,x,u) );
k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/1,u) );
f_ud = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );


%% ************************** Basis functions *****************************
basisFunction = 'rbf';
% RBF centers
Nrbf = 10; % 100 orig
cent = rand(n,Nrbf)*2 - 1; % -1 ??
rbf_type = 'thinplate';
% Lifting mapping - RBFs + the state itself
liftFun = @(xx)( [xx;rbf(xx,cent,rbf_type)] );
Nlift = Nrbf + n;


%% ************************** Collect data ********************************
% tic
disp('Starting data collection')
Nsim = 200; % 200 orig
Ntraj = 1000; % 1000 orig

% Random forcing
Ubig = 2*rand([Nsim m Ntraj]) - 1; % -1 ?? -->> for making normal distribution?!
% Ubig(:,1,:) = 0.1*Ubig(:,1,:);
% Ubig(:,2,:) = 2*Ubig(:,2,:);
% Ubig(:,3,:) = 5*Ubig(:,3,:);
% Ubig(:,4,:) = 100*Ubig(:,4,:);


% Random initial conditions
Xcurrent = (rand(n,Ntraj)*2 - 1);

X = []; Y = []; U = [];
for i = 1:Nsim
    u(:,:) = Ubig(i,:,:);
    Xnext = f_ud(0,Xcurrent,u);
    X = [X Xcurrent];
    Y = [Y Xnext];
    U = [U u];
    Xcurrent = Xnext;
end
% fprintf('Data collection DONE, time = %1.2f s \n', toc);


%% ******************************* Lift ***********************************

disp('Starting LIFTING')
% tic
Xlift = liftFun(X);
Ylift = liftFun(Y);
% fprintf('Lifting DONE, time = %1.2f s \n', toc);


%% ********************** Build predictor *********************************

disp('Starting REGRESSION')
% tic
W = [Ylift; X];
V = [Xlift; U];
VVt = V*V';
WVt = W*V';
M = WVt * pinv(VVt); % Matrix [A B; C 0]
Alift = M(1:Nlift,1:Nlift);
Blift = M(1:Nlift,Nlift+1:end);
Clift = M(Nlift+1:end,1:Nlift);

% fprintf('Regression done, time = %1.2f s \n', toc);


%% *********************** Predictor comparison ***************************

Tmax = 5;
Nsim = Tmax/deltaT;
% u_dt = @(i)((-1).^(round(i/30))); % control signal

% Initial condition
x0 = [0.5;0.1];
x_true = x0;

% Lifted initial condition
xlift = liftFun(x0);

U = ones([m Nsim]);
U(1,:) = 0.1*U(1,:);
U(2,:) = 2*U(2,:);
U(3,:) = 5*U(3,:);
U(4,:) = 1*U(4,:);

% Simulate
for i = 1:Nsim
    u = U(:,i)*1;%sin(0.05*i);
    
    % Koopman predictor
    xlift = [xlift, Alift*xlift(:,end) + Blift*u]; % Lifted dynamics -->> Masoud: which is linear!
    
    % True dynamics
    x_true = [x_true, f_ud(0,x_true(:,end),u) ];
    
end
x_koop = Clift * xlift; % Koopman predictions -->> Masoud: Just y = Cx


%% ****************************  Plots  ***********************************
lw = 3;
E2_1 = (x_true(1,:) - x_koop(1,:)).^2;
RMSE_1 = sqrt(mean(E2_1));
E2_2 = (x_true(2,:) - x_koop(2,:)).^2;
RMSE_2 = sqrt(mean(E2_2));

RMSE_avg = (RMSE_1 + RMSE_2)/2



% figure % prediction
% plot([0:Nsim]*deltaT,x_true(2,:),'linewidth',lw); hold on
% plot([0:Nsim]*deltaT,x_koop(2,:), '--r','linewidth',lw)
% title('Predictor comparison - $x_2$','interpreter','latex'); xlabel('Time [s]','interpreter','latex');
% set(gca,'fontsize',20)
% LEG = legend('True','Koopman','location','southwest');
% set(LEG,'interpreter','latex')
% 
% figure % prediction
% plot([0:Nsim]*deltaT,x_true(1,:),'linewidth',lw); hold on
% plot([0:Nsim]*deltaT,x_koop(1,:), '--r','linewidth',lw)
% title('Predictor comparison - $x_1$','interpreter','latex'); xlabel('Time [s]','interpreter','latex');
% set(gca,'fontsize',20)
% LEG = legend('True','Koopman','location','southwest');
% set(LEG,'interpreter','latex')
fprintf('Total time = %1.2f s \n', toc);
