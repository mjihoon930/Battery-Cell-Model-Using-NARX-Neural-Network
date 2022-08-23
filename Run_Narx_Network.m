clc
clear all
close all

%Train Data
file_name = 'FDM_N_50_(2).mat'; 
training_data = load(file_name); 

U = training_data.ans.I.signals.values';
u = num2cell(U);
y = training_data.ans.csn.signals.values';
csn = num2cell(y);

%Levenberg-Marquardt backpropagation
trainFcn = 'trainlm';

%define the input delays, feedback delays, and size of the hidden layers
inputDelays = 1:2;
feedbackDelays = 1:2;
hiddenLayerSize = 10;

%Nonlinear Autoegressive Network with External Input
narx_net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
%narx_net.divideFcn = '';
narx_net.trainParam.min_grad = 1e-100;

%Prepare the time series data
[x,xi,ai,target] = preparets(narx_net,u,{},csn);

%Setup for Training, Validation, Testing
% narx_net.divideParam.trainRatio = 70/100;
% narx_net.divideParam.valRatio = 15/100;
% narx_net.divideParam.testRatio = 15/100;

%Train the narx network
narx_net = train(narx_net,x,target,xi,ai);

view(narx_net)

%Test the narx network
Y = narx_net(x,xi,ai);
e = gsubtract(target,Y); %error
performance = perform(narx_net,target,Y);

%result for narx network 
csn1 = csn(1:20002);
u1 = u(1:20002);
[p1,Pi1,Ai1,t1] = preparets(narx_net,u1,{},csn1);
yp1 = narx_net(p1,Pi1,Ai1);
TS = 0.1:0.1:2000;

figure(1)
plot(TS,cell2mat(t1),'b',TS,cell2mat(yp1),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

%________________________________________________________________________________

%Closed Loop Network: Use this network to do multi-step prediction
%The function closeloop replaces the feedback input with a direct
%connection form the outout layer
%-> can not fully understand the closed loop network yet.

narx_net_closed = closeloop(narx_net);

narx_net_closed.name = [narx_net_closed.name 'Closed Loop'];
view(narx_net_closed)

%Prepare the time series data for Closed Loop Network
[xc,xic,aic,target_c] = preparets(narx_net_closed,u,{},csn);

%Test the closed loop narx network 
Y_c = narx_net_closed(xc,xic,aic);
e_c = gsubtract(target_c,Y_c); %error
performance_c = perform(narx_net_closed,target_c,Y_c);

%result #1
csn2 = csn(1:20002);
u2 = u(1:20002);
[p2,Pi2,Ai2,t2] = preparets(narx_net_closed,u2,{},csn2);
yp2 = narx_net_closed(p2,Pi2,Ai2);
TS = 0.1:0.1:2000;

figure(2)
plot(TS,cell2mat(t2),'b',TS,cell2mat(yp2),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

%________________________________________________________________________________

%Step-Ahead Prediction Network
%-> understand how it works and why this is used.

narx_net_rd = removedelay(narx_net);

%narx_net_rd.name = [narx_net_rd.name 'Remove Delay'];
%view(narx_net_rd)

%Prepare the time series data for Closed Loop Network
[xs,xis,ais,target_s] = preparets(narx_net_rd,u,{},csn);

%Test the step-ahead narx network
Y_s = narx_net_rd(xs,xis,ais);
e_s = gsubtract(target_s,Y_s); %error
performance_s = perform(narx_net_rd,target_s,Y_s);

%result for step-ahead prediction network
csn3 = csn(1:20001);
u3 = u(1:20001);
[p3,Pi3,Ai3,t3] = preparets(narx_net_rd,u3,{},csn3);
yp3 = narx_net_rd(p3,Pi3,Ai3);
TS3 = 0.1:0.1:2000;

figure(3)
plot(TS3,cell2mat(t3),'b',TS3,cell2mat(yp3),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

%________________________________________________________________________________
%Test the narx battery model
%Test the narx model using other input current profile

%Test Data
file_name_test = 'FDM_N_50_validation_1.mat'; 
test_data = load(file_name_test); 

U_test = test_data.ans.I.signals.values';
u_test = num2cell(U_test);
y_test = test_data.ans.csn.signals.values';
csn_test = num2cell(y_test);

%Prepare the time series data to test the Narx model
[x_t,xi_t,ai_t,target_t] = preparets(narx_net,u_test,{},csn_test);

Y_test = narx_net(x_t,xi_t,ai_t);
e_c = gsubtract(target_t,Y_test); %error
performance_t = perform(narx_net,target_t,Y_test);

%test result
csn4 = csn_test(1:20002);
u4 = u_test(1:20002);
[p4,Pi4,Ai4,t4] = preparets(narx_net,u4,{},csn4);
yp4 = narx_net(p4,Pi4,Ai4);
TS4 = 0.1:0.1:2000;

figure(4)
plot(TS4,cell2mat(t4),'b',TS4,cell2mat(yp4),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')
