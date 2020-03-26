classdef neuralNetwork < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
       Wt = truncate(makedist('Normal'), -0.99, 0.99);
       inodes = 0; % ��������� ����� �������� ����
       hnodes = 0; % ��������� ����� �������� ����
       onodes = 0; % ���������� ����� ��������� ����
       lr = 0;     % ����������� �������� ����
       W_input_hidden;
       W_hidden_output;
    end
    methods (Access = private)
        function y = sigmoida(x)
            %SIGMOIDA Summary of this function goes here
            %   Detailed explanation goes here
            y = 1./(1+exp(-x));
        end
    end
    methods (Access = public)
        % ������������� ��������� ����
        function obj = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
            obj.inodes = inputnodes;
            obj.hnodes = hiddennodes;
            obj.onodes = outputnodes;
            obj.lr = learningrate;
            
            %% ���������� ������������� � ��������� �� -0.99 �� 0.99
%            obj.W_input_hidden = random(obj.Wt,obj.hnodes,obj.inodes);
%            obj.W_hidden_output = random(obj.Wt,obj.onodes,obj.hnodes);
%             obj.W_input_hidden = random(obj.Wt,obj.inodes,obj.hnodes);
%             obj.W_hidden_output = random(obj.Wt,obj.hnodes,obj.onodes);
           
            %% ���������� �������� ������������� ������� �����. ���������� �������������, ��� mean = 0, sigma  = 1/(sqrt(nodes))   
           obj.W_input_hidden = normrnd(0.0, power(obj.hnodes,-0.5),obj.hnodes,obj.inodes);
           obj.W_hidden_output = normrnd(0.0, power(obj.onodes,-0.5),obj.onodes,obj.hnodes);
            
            %% �������� ��������
%           obj.W_input_hidden = [[0.9; 0.2; 0.1;] [0.3; 0.8; 0.5] [0.4; 0.2; 0.6]]; % ������� ������������ ��� �������� (����������� ����) ����
%           obj.W_hidden_output = [[0.3; 0.6; 0.8] [0.7; 0.5; 0.1] [0.5; 0.2; 0.9]]; % ������� ������������ � ������ �������� ����� ����
  
        end
        % �������� �������� ����
        function obj = train(obj, I, target_list)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            %obj.Property1 = inputArg1 + inputArg2;
            
             hidden_inputs = obj.W_input_hidden*I;
             hidden_outputs = sigmoida(hidden_inputs);
             
             final_inputs = obj.W_hidden_output*hidden_outputs;
             final_outputs = sigmoida(final_inputs);
             
             output_errors = target_list - final_outputs;
            
             hidden_errors = (obj.W_hidden_output')*output_errors;
             
             obj.W_hidden_output = obj.W_hidden_output + obj.lr * ...
             ((output_errors .* final_outputs .* (1.0 - final_outputs))*(hidden_outputs'));
             
             obj.W_input_hidden = obj.W_input_hidden + obj.lr * ...
             ((hidden_errors .* hidden_outputs .* (1.0 - hidden_outputs))*(I'));
             
        end
        function out = getW_hidden_output(obj)
            out = obj.W_hidden_output;
        end
        function out = getW_input_hidden(obj)
            out = obj.W_input_hidden;
        end
        % ����� ��������� ����
        function out = query(obj, I)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            %outputArg = obj.Property1 + inputArg;
            hidden_inputs = obj.W_input_hidden*I; % ��������� �������� ������� ���� (������ ������� ����� �������� ����)

            hidden_outputs = sigmoida(hidden_inputs); % ���������� ������� ��������� ������� ��� ������, ����������� � ������ 1-�� ����
            
            final_inputs = obj.W_hidden_output*hidden_outputs; % ��������� �������� ���������(����������) ����

            final_outputs = sigmoida(final_inputs); % ���������� ������� ��������� �� ������, ����������� � ���������(����������) ����

            out = final_outputs;
        end
    end
end

