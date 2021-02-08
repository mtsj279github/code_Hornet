 n = 100;
p = 0.01;
load('Latitude');
load('Longitude');
z = zeros(n);
% Se = rand(n)<p;
Latitude = Latitude*1000-48000;
Longitude = -Longitude*1000-122000;
a = sum(Latitude);
b = sum(Longitude);
Latitude = Latitude/sum(Latitude)*100;
Longitude = Longitude/sum(Longitude)*100;
Se = zeros(n);
for i = 1:14
    Se(round(Latitude(i)),round(Longitude(i))) = 1;
end
Sd = zeros(n+2);
Ph = image(cat(3,Se,z,z));
x = 5;
while(x)
    x= x-1;
    Sd(2:n+1,2:n+1) = Se;
    Sum = Sd(1:n, 2:n+1)+ Sd(3:n+2, 2:n+1)+ Sd(2:n+1, 1:n) + Sd(2:n+1, 3:n+2);
    
    for i = 1:n
    for j = 1:n
    if mod(Sum(i,j),2) ==1 
        Se(i,j) = 1;
    else
        Se(i,j) = 0;
    end
    end
    end
    set(Ph,'cdata',cat(3,Se,z,z));
    drawnow;
end

m = [];
y = [];
x =1;
for i = 1:n
    for j = 1:n
        if(Se(i,j)==1)
            m(x) = i;
            y(x) = j;
            x = x+1;
        end
    end
end
m = (m/100*a+48000)/1000;
y = -(y/100*b+122000)/1000;
figure(1);
plot(m);
figure(2);
plot(y);