density = [300, 600, 900, 1200, 1500, 1800, 2100];
for d = 1:1
    for it_ = 1:1
        Tsim = 6000000; %simulation time
        volumn = density(3); %traffic flow on one lane 
        flagPoisson = true; %distribution
        random = false;
        if flagPoisson
            t=0;N=0;
            arvTimeNewVeh=zeros(int64(volumn*Tsim/3600+10),12);
            while(min(t)<=Tsim)
                U=rand(1, 12);
                if mod(N, 10) ==0 && random
                    [a, seq] = max(rand(1,7));
                    volumn = density(seq);
                end
                % seq = min(seq + 2, 7);
                lamda=volumn/3600;
                elapsedTime=-1/lamda*log(U);
                elapsedTime(elapsedTime<1) = 1; % guarantee the time gap
                t=t+elapsedTime;
                N=N+1;
                arvTimeNewVeh(N,:)=t;
            end
        end
        arvTimeNewVeh(end +1, :) = zeros([1,12]);
        save("arvTimeNewVeh_new_" + num2str(volumn) + "_multi3_3_l" + ".mat", 'arvTimeNewVeh'); 
%         save("arvTimeNewVeh_new_random_density_single_intersection.mat", 'arvTimeNewVeh'); 
    end
end