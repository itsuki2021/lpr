docker run -d --rm --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --net=host \
        -v "$(dirname "$(pwd)")"/torchserve/model-store/:/home/model-server/model-store \
        -v "$(dirname "$(pwd)")"/torchserve/config.properties:/home/model-server/config.properties \
        -v "$(dirname "$(pwd)")"/torchserve/dict_printed_chinese_alpha_lp.txt:/home/model-server/dict_printed_chinese_alpha_lp.txt \
        -v "$(dirname "$(pwd)")"/torchserve/dockerd-entrypoint.sh:/home/model-server/dockerd-entrypoint.sh \
        --entrypoint /home/model-server/dockerd-entrypoint.sh \
        stliu2022/mmocrserve:0.4.1-cpu serve
