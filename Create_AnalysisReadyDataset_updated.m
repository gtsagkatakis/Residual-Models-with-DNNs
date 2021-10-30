close all
clearvars
clc

grid_on_vertices=[-8.00 -5.00 176.22;
-7.00 -5.00 176.23;
-6.00 -5.00 176.14;
-5.00 -5.00 177.12;
-4.00 -5.00 178.93;
-3.00 -5.00 179.94;
-8.00 -4.40 175.35;
-7.00 -4.40 174.97;
-6.00 -4.40 175.45;
-5.00 -4.40 177.20;
-4.00 -4.40 180.48;
-3.00 -4.40 183.09;
-8.00 -3.80 194.43;
-7.00 -3.80 195.93;
-6.00 -3.80 194.29;
-5.00 -3.80 196.62;
-4.00 -3.80 206.67;
-3.00 -3.80 215.18;
-8.00 -3.20 174.71;
-7.00 -3.20 175.64;
-6.00 -3.20 176.27;
-5.00 -3.20 183.56;
-4.00 -3.20 213.42;
-3.00 -3.20 244.48;
-8.00 -2.60 283.06;
-7.00 -2.60 284.81;
-6.00 -2.60 295.27;
-5.00 -2.60 339.80;
-4.00 -2.60 479.93;
-3.00 -2.60 618.21;
-8.00 -2.00 344.12;
-7.00 -2.00 349.83;
-6.00 -2.00 363.95;
-5.00 -2.00 437.36;
-4.00 -2.00 648.71;
-3.00 -2.00 845.31];

grid_on_vertices_n=grid_on_vertices;
grid_on_vertices_n(:,3)=grid_on_vertices_n(:,3)/max(grid_on_vertices(:,3));


addpath(genpath('jsonlab-1.9'))

dir_list{1}='D:\AstroWork_w_GV\Astro_images_24_6_2020\images_analytic\';
dir_list{2}='D:\AstroWork_w_GV\Astro_images_24_6_2020\images_merger\';
dir_list{3}='D:\AstroWork_w_GV\Astro_images_24_6_2020\images_spiral\';

num_examples=90000;

I=zeros(num_examples,100,100);
LBL=zeros(num_examples,4);
LBL_q=zeros(num_examples,1);
LBL_points=zeros(num_examples,2);

LBL_a_dist=zeros(num_examples,50);
LBL_b_dist=zeros(num_examples,50);

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
LBL_type=zeros(num_examples,1);
LBL_msk=zeros(num_examples,1);

exp_idx=1;

min_dst_mtr=@(x,y,x0,y0) abs(x-x0)+abs(y-y0);

for mm=1:numel(dir_list)
    dir_name=dir_list{mm};
    ls=dir([dir_name,'img*']);
    
    for tt=1:numel(ls)
        [mm/numel(dir_list),tt/numel(ls)]
        
        fname=[dir_name,ls(tt).name,'\img.fits'];
        data = fitsread(fname);
        for ww=1:3
            switch ww
                case 1
                    fname2=[dir_name,ls(tt).name,'\mask_00.fits'];
                    msk_type=0;
                case 2
                    fname2=[dir_name,ls(tt).name,'\mask_01.fits'];
                    msk_type=1;
                case 3
                    fname2=[dir_name,ls(tt).name,'\mask_02.fits'];
                    msk_type=2;
            end
            msk_out=fitsread(fname2);
            
            
            fname3=[dir_name,ls(tt).name,'\pars.json'];
            specs=loadjson(fname3);
            
            img=data.*msk_out;
            for qq=1:4
                r_img=imrotate(img,(qq-1)*90);
                I(exp_idx,:,:)=r_img;
                a_up=specs.a_dist_1;
                a_lw=specs.a_dist_2;
                
                b_up=specs.b_dist_1;
                b_lw=specs.b_dist_2;
                
                
                LBL(exp_idx,:)=[a_up,a_lw,b_up,b_lw];
                LBL_points(exp_idx,:)=[specs.a0,specs.b0];
                
                [~,ra]=min(abs(LUT_a_50bin-specs.a0));
                [~,rb]=min(abs(LUT_b_50bin-specs.b0));
                
                LBL_a_dist(exp_idx,:)=LUT_a_dist(LUT_a(ra),:);
                LBL_b_dist(exp_idx,:)=LUT_b_dist(LUT_b(rb),:);
                
                LBL_a_q(exp_idx)=ra;
                LBL_b_q(exp_idx)=rb;
                LBL_type(exp_idx)=mm;
                LBL_msk(exp_idx)=msk_type;
                
                [~,tmp_b]=min(min_dst_mtr(specs.a0,specs.b0,grid_on_vertices(:,2),grid_on_vertices(:,1)));
                
                N=30;
%                 P = 1 / (1 + exp(ra/50+rb/50));
                P = 1 / (1 + exp(2*grid_on_vertices_n(tmp_b,3)));

                P_lst(exp_idx)=P;

%                 rnd_shift_left=binornd(10,0.5);
%                 rnd_shift_right=binornd(10,0.5);
                rnd_shift_left=binornd(N,P);
                rnd_shift_right=binornd(N,P);

                tmp=max(1,(ra-rnd_shift_right)):min(50,(ra+rnd_shift_left));
                tmp2=zeros(50,1);
                tmp2(tmp)=1;
                tmp2=tmp2./sum(tmp2);
                
                LBL_a_dist_BRN(exp_idx,:)=tmp2;
                
%                 rnd_shift_left=binornd(10,0.5);
%                 rnd_shift_right=binornd(10,0.5);
                rnd_shift_left=binornd(N,P);
                rnd_shift_right=binornd(N,P);
                
                tmp=max(1,(rb-rnd_shift_right)):min(50,(rb+rnd_shift_left));
                tmp2=zeros(50,1);
                tmp2(tmp)=1;
                tmp2=tmp2./sum(tmp2);
                
                LBL_b_dist_BRN(exp_idx,:)=tmp2;
                
                exp_idx=exp_idx+1;
                
            end
            
        end
        
    end
    
end


I2=single(I);

I_part1=I2(1:30000,:,:);
I_part2=I2(30001:60000,:,:);
I_part3=I2(60001:90000,:,:);

save('Data_in_mat\images_May2020_AllModels_30K_dist_part1_analytic_newUnc_newPS.mat','I_part1');
save('Data_in_mat\images_May2020_AllModels_30K_dist_part2_merger_newUnc_newPS.mat','I_part2');
save('Data_in_mat\images_May2020_AllModels_30K_dist_part3_spiral_newUnc_newPS.mat','I_part3');

% save('Data_in_mat\labels_May2020_AllModels_30K_dist_part1_analytic.mat','LBL_a_dist','LBL_b_dist','LBL_points','LBL_a_q','LBL_b_q');
