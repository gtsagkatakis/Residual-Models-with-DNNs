close all
clearvars
clc

addpath(genpath('jsonlab-1.9'));

dir_name='residuals_July2019\';

ls=dir(dir_name);
idx=1;
figure;

num_examples=24000;

I=zeros(num_examples,100,100);
LBL=zeros(num_examples,4);
LBL_q=zeros(num_examples,1);
LBL_points=zeros(num_examples,2);

LBL_a_dist=zeros(num_examples,50);
LBL_b_dist=zeros(num_examples,50);

rand_lst=randperm(numel(ls)-2)+2;
rand_lst2=randperm(num_examples);

LUT_a_dist=zeros(5,50);
LUT_a_dist(1,1:10)=0.1;
LUT_a_dist(2,11:20)=0.1;
LUT_a_dist(3,21:30)=0.1;
LUT_a_dist(4,31:40)=0.1;
LUT_a_dist(5,41:50)=0.1;
LUT_a_50bin=[-5:0.06:-2.05];
LUT_a=[ones(10,1)*1;ones(10,1)*2;ones(10,1)*3;ones(10,1)*4;ones(10,1)*5];


LUT_b_dist=zeros(5,50);
LUT_b_dist(1,1:10)=0.1;
LUT_b_dist(2,11:20)=0.1;
LUT_b_dist(3,21:30)=0.1;
LUT_b_dist(4,31:40)=0.1;
LUT_b_dist(5,41:50)=0.1;
LUT_b_50bin=-8:0.1:-3.05;
LUT_b=[ones(10,1)*1;ones(10,1)*2;ones(10,1)*3;ones(10,1)*4;ones(10,1)*5];


LBL_a_q=zeros(num_examples,1);
LBL_b_q=zeros(num_examples,1);

for tt=1:2000      %numel(ls)
    tt
    zz=rand_lst(tt);
    
    fname=[dir_name,ls(zz).name,'\res.fits'];
    data = fitsread(fname);
    for ww=1:3
        switch ww
            case 1
                fname2=[dir_name,ls(zz).name,'\mask_00.fits'];
            case 2
                fname2=[dir_name,ls(zz).name,'\mask_01.fits'];
            case 3
                fname2=[dir_name,ls(zz).name,'\mask_02.fits'];
        end
        data_flt=fitsread(fname2);
        fname3=[dir_name,ls(zz).name,'\pars.json'];
        specs=loadjson(fname3);
        
        img=data.*data_flt;
        for qq=1:4
            r_img=imrotate(img,(qq-1)*90);
            I(rand_lst2(idx),:,:)=r_img;
            a_up=specs.a_dist_1;
            a_lw=specs.a_dist_2;
            
            b_up=specs.b_dist_1;
            b_lw=specs.b_dist_2;
            
            LBL(rand_lst2(idx),:)=[a_up,a_lw,b_up,b_lw];
            LBL_points(rand_lst2(idx),:)=[specs.a0,specs.b0];

            [~,ra]=min(abs(LUT_a_50bin-specs.a0));
            [~,rb]=min(abs(LUT_b_50bin-specs.b0));
            
            LBL_a_dist(rand_lst2(idx),:)=LUT_a_dist(LUT_a(ra),:);
            LBL_b_dist(rand_lst2(idx),:)=LUT_b_dist(LUT_b(rb),:);
            
            LBL_a_q(rand_lst2(idx))=ra;
            LBL_b_q(rand_lst2(idx))=rb;
            
            idx=idx+1;
            
        end
    end
    
end

rand_lst2_temp=randperm(max(rand_lst2));

figure;
for idx=1:100
    clf; plot( LBL_b_dist(rand_lst2_temp(idx),:))
    hold on; scatter( LBL_b_q(rand_lst2_temp(idx)),0.1,'x')
    legend('target distribution','ground truth')
    pause(0.5);
end

% save('images_Feb2020_20000_dist.mat','I');
% save('labels_Feb2020_20000_dist.mat','LBL_a_dist','LBL_b_dist','LBL_points','LBL_a_q','LBL_b_q');

return


