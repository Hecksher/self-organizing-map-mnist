%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Trabalho número 2 de Sistemas Inteligentes.
% Código por Pedro Henrique Faber e Lucas Guilhem de Matos
% Mapa de Kohonen para classificação de dados da base MNIST.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inicialização do Programa.

clear all
close all
clc 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Vetorização das Imagens de Treinamento e Validação. 
%  NÃO ALTERAR ESSE TRECHO

% Imagens para treinamento.
treino = loadMNISTImages('train-images.idx3-ubyte');

%Labels para o Treinamento
labels = loadMNISTLabels('train-labels.idx1-ubyte');
num = 5000;%length(labels); %Número de Labels, consequentemente, de Imagens.

%Imagens para validação.
teste = loadMNISTImages('t10k-images.idx3-ubyte');

%Labels para validação.
labels_teste = loadMNISTLabels('t10k-labels.idx1-ubyte');
num2 = 5000;%length(labels_teste); %Número de Labels, consequentemente, de Imagens.

%Alem da definição de Matriz de Respostas
%Matriz com as respostas. As referências são 0.1 e 0.9. É comum essa
%Utilização para que o erro seja reduzido.
answers = 0.1*ones(num,10); 
%Uma segestão comum é utilizar 0.1 e 0.9
for i=1:num
    answers(i,labels(i)+1) = 0.9;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Definição dos Parâmetros do Mapa de Kohonen. Aprendizado, Raio e Pesos.

% Tamanho do mapa.
N = 30;

% Pesos dos neurios por tamanho de entrada
w = rand(784 , N , N);

% O Mapa de Kohonen. Matriz onde os neurônios serão rotulados.
kohonen = zeros(N , N);

% Taxa de Aprendizado Inicial. Esse valor é alterado ao longo da execução. 
eta_ini= 0.9;
eta = eta_ini;

% Para o caso de utilização de uma gaussiana na função de vizinhança.
sigma_ini = N/3;
sigma = sigma_ini;

%Variáveis de Performance
acertos = 0;
erros = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Definição dos Parâmetros para o Perceptron.

%Perceptron multicamada para identificação de imagens.
%Valor inicial da primeira camada. 
cam1 = 40;
%Valor inicial da segunda Camada. 
cam2 = 100;
%Valor inicial da terceira Camada.
cam3 = 10; %Essa camada não deve ser modificada
%Metodologia de Aprendizado com o Critério de Mínimos Quadrados.
%Função de Ativação A ser Definica pelo Usuário

%Definição do Número máximo de Épocas que serão utilizadas em treinamento. Uma
%época é a utilização de todas as imagens de treinamento. O número de épocas,
%é o número de vezes que o programa leu todas as imagens.
epoca=5;

%Declaração das Matrizes de pesos Inicial - Pesos Aleatórios
w1 = rand(N^2 + 1 ,cam1)*.2-.1; %Primeira camada
w2 = rand(cam1+1,cam2)*.2-.1; %Segunda camada
w3 = rand(cam2+1,cam3)*.2-.1;%Terceira Camada

eta_p = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Laços Principais

tic
%% Laço de Treinamento do Mapa de Kohonen
for k = 1:num

    %Laço para a verificação de qual neurônio responde à entrada. em
    %questão.
    for i = 1:N
        for j = 1:N

            erro(:,i,j) = treino(:,k) - w(:,i,j); %Verifica as distância entre vetores
            distance(i,j) = norm(erro(:,i,j)); % Calcula a distancia euclidiana, ou a norma.

        end
    end

    [M,I] = min(distance(:));
    [min_i, min_j] = ind2sub(size(distance),I); %Verifica qual o �?ndice do neurônio responde melhor à entrada.


    % ROTULA O NEURÔNIO NO MAPA DE KOHONEN    
    kohonen(min_i, min_j) = labels(k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %ATUALIZAÇÃO DOS PESOS. 

    % Esse trecho define os limites de atuação da atualização de pesos
    % baseando-se no raio de Vizinhança. Função Gaussiana

   % Atualização dos Pesos do Mapa.
   % A sintaxe é w(k+1) = w(k) + (Taxa de Aprendizado)*(Função de
   % Vizinhança)*(x(k)-w(k)); A Função de Vizinhança é uma Gaussiana.

    for i = 1:N
        for j = 1:N

            w(: , i, j) = w(:, i, j) + exp(-(norm([i,j]-[min_i,min_j])^2)/(2*(sigma^2)))*eta*erro(:,i,j);

        end
    end

    % Atualização da Variância da Gaussiana
    plot1(k) = sigma; %Vetor para acompanhar a evolução do Raio de Vizinhança.
    % O 0.75 na euqação existe para que o raio de vizinhança decaia mais
    % rapidamente. de forma que a variância seja
    sigma = sigma_ini*exp(-k/(0.75*num/(4.6)));

    % Atualição da Taxa de Aprendizado.
    plot2(k) = eta; %Vetor para acompanhar a taxa de aprendizado.
    eta = eta_ini*exp(-k/(num/4.5)); 

end
toc
% tic
% %% Laço de Validação Usando Mapa de Kohonen.
% for k = 1:num2
% 
%     %Laço para a verificação de qual neurônio responde à entrada. em
%     %questão.
%     for i = 1:N
%         for j = 1:N
% 
%             erro(:,i,j) = teste(:,k) - w(:,i,j); %Verifica as distância entre vetores
%             distance(i,j) = norm(erro(:,i,j)); % Calcula a distancia euclidiana, ou a norma.
% 
%         end
%     end
% 
%     [M,I] = min(distance(:));
%     [min_i, min_j] = ind2sub(size(distance),I); %Verifica qual o �?ndice do neurônio responde melhor à entrada.
% 
%     if kohonen(min_i,min_j) == labels_teste(k)
%         acertos = acertos+1;
%     else
%         erros = erros+1;
%     end
% end
% toc

%% Laço de Treinamento e validação do Perceptron.
for j=1:epoca
    %Loop de Propagação do erro para uma época
    for i=1:num
        
        % Cria a Entrada para O treinamento
        for n = 1:N
            for m = 1:N

            erro(:,n,m) = treino(:,i) - w(:,n,m); %Verifica as distância entre vetores
            distance(n,m) = 1/norm(erro(:,n,m)); % Calcula a distancia euclidiana, ou a norma.

            end
        end
                               
        s1 = [distance(:)' 1]*w1;
        y1 = 1./(1+exp(-s1));
        s2 = [y1 1]*w2;
        y2 = 1./(1+exp(-s2));
        s3 = [y2 1]*w3;
        y3 = 1./(1+exp(-s3));

        %Parâmetros a serem avaliados
        %Evolução do erro quadrático
        %sse(i)=sse(i)+(d(j)-y2)^2;

        %Retropropagação
        %A derivada da função Logística é g(s)*(1-g(s)), como no caso a
        %função g(s) é igual a y2, temos y2*(1-y2)
        %O erro é dado por (d-y2)*g'(s) = (d-y2)*y2*(1-y2)
        %Para a segunda camada escondida.
        e3=(answers(i,:)-y3).*y3.*(1-y3); 
        dw3=transpose([y2 1])*e3;

        %O Cálculo geral do perceptron é dw = eta * (d-y) * "entrada"
        %Para a primeira camada escondida
        e2= (e3*transpose(w3(1:cam2,1:cam3))).*y2.*(1-y2); % Propagação do erro para 1a camada
        dw2=transpose([y1 1])*e2;

        %O Cálculo geral do perceptron é dw = eta * (d-y) * "entrada"
        %Para a camada de entrada
        e1= (e2*transpose(w2(1:cam1,1:cam2))).*y1.*(1-y1); % Propagação do erro para 1a camada
        dw1=[distance(:); 1]*e1;

        %Atualização dos pesos depois do treinamento.
        w1=w1+eta_p*dw1; 
        w2=w2+eta_p*dw2;
        w3=w3+eta_p*dw3;

        %Atualização da taxa de aprendizado. Indicado em alguns casos
        %eta = 0.9999*eta;

        %Cálculo do erro quadrático médio. Medição de Desempenho.
        sse(i) = ((answers(i,:)-y3)*transpose(answers(i,:)-y3))/cam3;
        
    end
    
    %Erro quadrático médio no treinamento.
    eqm(j) = (sum(sse))/num;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Constantes para a nálise da porcentagem de acertos.
    acertos = 0;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Loop de Teste. Esse Loop É utilizado para a verificação do índice de
    %acertos depois do treinamento

    for i=1:num2
        
        for n = 1:N
            for m = 1:N

            erro(:,n,m) = teste(:,i) - w(:,n,m); %Verifica as distância entre vetores
            distance(n,m) = 1/norm(erro(:,n,m)); % Calcula a distancia euclidiana, ou a norma.

            end
        end

           %Cálculo da saída para as imagens de validação
        s1 = [distance(:)' 1]*w1;
        y1 = 1./(1+exp(-s1));
        s2 = [y1 1]*w2;
        y2 = 1./(1+exp(-s2));
        s3 = [y2 1]*w3;
        y3 = 1./(1+exp(-s3));

        [M,I] = max(y3(:));% Essa função acha o índice do maior valor do vetor.

        %Atualização de Valores de Acertos e Erros.
        %Notar que a label começa no zero, mas o índice do vetor começa em 1,
        %então é importante somar 1 ao valor da label para realizar a
        %comparação.
        if I==(labels_teste(i)+1)
            acertos = acertos+1;
        end
    end

    %Cálculo do índice de acerto. Número de Acertos sobre o total de Imagens
    %vistas.
    indice_acerto(j) = 100*(acertos/num2);
    indice_erro(j) = 100-indice_acerto(j);
    j
end

figure
surf(kohonen);
set(gca,'FontSize',25)

