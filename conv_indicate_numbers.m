
% «агружаю картинки чисел дл€ »ндикации в массив (картинки должны быть
% одинакого размера)
init_result = 0;

for i = 0:9
    limage = import_img(strcat('numbers_elite\',strcat(num2str(i),'.bmp')));

    [xim,yim] = size(limage);
    if(init_result == 0)
        elite_numbers = zeros(length(0:9),(xim*yim) + 1);
        init_result = 1;
    end

    r = reshape(limage,[],xim*yim);
    elite_numbers((i+1),:) = [i r];
end


