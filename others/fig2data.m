'''
use this in matlab and obtain data of points in a given figure
'''

n = 5; % the number of points
max_x = 256; % the max of x label
min_x = 16 ; % the min of x label
max_y = 0.4; % the max of y label
min_y = 0.3; % the min of y label
imshow('1.png')
disp('鼠标单击原点')
[x,y] = ginput(1); 
origin_y=y;
origin_x=x;
disp('鼠标单击原点对角线的那个点')
[x,y] = ginput(1); 
fin_y=y;
fin_x=x;
xlabel =(max_x-min_x)/ abs(fin_x-origin_x);
ylabel = (max_y-min_y)/abs(fin_y-origin_y);
disp('依次单击想取出来的点')
[x,y]=ginput(n);
for i=1:1:n
  capture_x1(i) = abs(x(i)-origin_x)*xlabel + min_x;
  capture_y1(i) = abs(y(i)-origin_y)*ylabel + min_y;
end
 
disp('点的结果存在capture_x1和capture_y1中')
 plot( capture_x1, capture_y1,'r')
 xlim([min_x max_x])
 ylim([min_y max_y])
grid minor
