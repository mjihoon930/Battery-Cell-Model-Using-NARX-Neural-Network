clc
clear all
close all

%Train Data

%Pulse input
file_name_1 = 'FDM_N_50_(2).mat'; 
training_data = load(file_name_1); 

U_1 = training_data.ans.I.signals.values';
u_1 = num2cell(U_1);
y_1 = training_data.ans.csn.signals.values';
csn_1 = num2cell(y_1);

%Sinusoid wave input
file_name_2 = 'FDM_N_50_fs.mat'; 
training_data = load(file_name_2); 

U_2 = training_data.ans.I.signals.values';
u_2 = num2cell(U_2);
y_2 = training_data.ans.csn.signals.values';
csn_2 = num2cell(y_2);

u = catsamples(u_1,u_2,'pad');
csn = catsamples(csn_1,csn_2,'pad');


%Levenberg-Marquardt backpropagation
trainFcn = 'trainlm';

%define the input delays, feedback delays, and size of the hidden layers
inputDelays = 1:3;
feedbackDelays = 1:1;
hiddenLayerSize = 10;

%Nonlinear Autoegressive Network with External Input
narx_net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
narx_net.trainParam.min_grad = 1e-100;
narx_net.trainParam.goal = 0;

%Prepare the time series data
[x,xi,ai,target] = preparets(narx_net,u,{},csn);

%Setup for Training, Validation, Testing
% narx_net.divideParam.trainRatio = 70/100;
% narx_net.divideParam.valRatio = 15/100;
% narx_net.divideParam.testRatio = 15/100;

%Train the narx network
narx_net = train(narx_net,x,target,xi,ai);

%view(narx_net)

%Test the narx network
Y = narx_net(x,xi,ai);
e = gsubtract(target,Y); %error
performance = perform(narx_net,target,Y);


%result for narx network 
csn1_1 = csn_1(1:20002);
u1_1 = u_1(1:20002);
[p1_1,Pi1_1,Ai1_1,t1_1] = preparets(narx_net,u1_1,{},csn1_1);
yp1_1 = narx_net(p1_1,Pi1_1,Ai1_1);
TS = 0.1:0.1:size(yp1_1,2)/10;

figure(1)
plot(TS,cell2mat(t1_1),'b',TS,cell2mat(yp1_1),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

csn1_2 = csn_2(1:20002);
u1_2 = u_2(1:20002);
[p1_2,Pi1_2,Ai1_2,t1_2] = preparets(narx_net,u1_2,{},csn1_2);
yp1_2 = narx_net(p1_2,Pi1_2,Ai1_2);
TS = 0.1:0.1:size(yp1_2,2)/10;

figure(2)
plot(TS,cell2mat(t1_2),'b',TS,cell2mat(yp1_2),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

%________________________________________________________________________________

%Closed Loop Network: Use this network to do multi-step prediction
narx_net_closed = closeloop(narx_net);

%narx_net_closed.name = [narx_net_closed.name 'Closed Loop'];
%view(narx_net_closed)

%Prepare the time series data for Closed Loop Network
[xc,xic,aic,target_c] = preparets(narx_net_closed,u,{},csn);

%Test with the closed loop narx network 
Y_c = narx_net_closed(xc,xic,aic);
e_c = gsubtract(target_c,Y_c); %error
performance_c = perform(narx_net_closed,target_c,Y_c);

%result
csn2_1 = csn_1(1:20002);
u2_1 = u_1(1:20002);
[p2_1,Pi2_1,Ai2_1,t2_1] = preparets(narx_net_closed,u2_1,{},csn2_1);
yp2_1 = narx_net_closed(p2_1,Pi2_1,Ai2_1);
TS2 = 0.1:0.1:size(yp2_1,2)/10;

figure(3)
plot(TS2,cell2mat(t2_1),'b',TS2,cell2mat(yp2_1),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

csn2_2 = csn_2(1:20002);
u2_2 = u_2(1:20002);
[p2_2,Pi2_2,Ai2_2,t2_2] = preparets(narx_net_closed,u2_2,{},csn2_2);
yp2_2 = narx_net_closed(p2_2,Pi2_2,Ai2_2);
TS2 = 0.1:0.1:size(yp2_2,2)/10;
 
figure(4)
plot(TS2,cell2mat(t2_2),'b',TS2,cell2mat(yp2_2),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

%________________________________________________________________________________

%Step Ahead Prediction

narx_net_rd = removedelay(narx_net);

%narx_net_rd.name = [narx_net_rd.name 'Remove Delay'];
%view(narx_net_rd)

%Prepare the time series data for Step Ahead Prediction
[xs,xis,ais,target_s] = preparets(narx_net_rd,u,{},csn);

%Test with the step ahead prediction
Y_s = narx_net_rd(xs,xis,ais);
e_s = gsubtract(target_s,Y_s); %error
performance_s = perform(narx_net_rd,target_s,Y_s);

%result for step ahead prediction network
csn3_1 = csn_1(1:20001);
u3_1 = u_1(1:20001);
[p3_1,Pi3_1,Ai3_1,t3_1] = preparets(narx_net_rd,u3_1,{},csn3_1);
yp3_1 = narx_net_rd(p3_1,Pi3_1,Ai3_1);
TS3 = 0.1:0.1:size(yp3_1,2)/10;

figure(5)
plot(TS3,cell2mat(t3_1),'b',TS3,cell2mat(yp3_1),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

csn3_2 = csn_2(1:20001);
u3_2 = u_2(1:20001);
[p3_2,Pi3_2,Ai3_2,t3_2] = preparets(narx_net_rd,u3_2,{},csn3_2);
yp3_2 = narx_net_rd(p3_2,Pi3_2,Ai3_2);
TS3 = 0.1:0.1:size(yp3_2,2)/10;

figure(6)
plot(TS3,cell2mat(t3_2),'b',TS3,cell2mat(yp3_2),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

%________________________________________________________________________________
%Test the narx battery model
%Test the narx model using other input current profile

%Test Data
file_name_test = 'FDM_N_50_D_SOC90_10_fs.mat'; 
test_data = load(file_name_test); 

U_test = test_data.ans.I.signals.values';
u_test = num2cell(U_test);
y_test = test_data.ans.csn.signals.values';
csn_test = num2cell(y_test);

%Prepare the time series data to test the Narx model
[x_t,xi_t,ai_t,target_t] = preparets(narx_net,u_test,{},csn_test);

Y_test = narx_net(x_t,xi_t,ai_t);
e_t = gsubtract(target_t,Y_test); %error
performance_t = perform(narx_net,target_t,Y_test);

%test result
csn4 = csn_test(1:35002);
u4 = u_test(1:35002);
[p4,Pi4,Ai4,t4] = preparets(narx_net,u4,{},csn4);
yp4 = narx_net(p4,Pi4,Ai4);
TS4 = 0.1:0.1:size(yp4,2)/10;

figure(7)
plot(TS4,cell2mat(t4),'b',TS4,cell2mat(yp4),'r','LineWidth',2)
legend('Target','Predicted value')
xlabel('time [t]')
ylabel('c_s_,_n [mol/cm^3]')

