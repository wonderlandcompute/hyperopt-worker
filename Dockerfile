FROM alpine:3.6

ENV CONDA_DIR=/opt/conda CONDA_VER=4.3.31
ENV PATH=$CONDA_DIR/bin:$PATH SHELL=/bin/bash LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN apk --update  --repository http://dl-4.alpinelinux.org/alpine/edge/community add \
    bash \
    build-base \
    bzip2 \
    curl \
    ca-certificates \
    git \
    glib \
    libstdc++ \
    libxext \
    libxrender \
    tini \
    unzip \
    && curl -L "https://github.com/andyshinn/alpine-pkg-glibc/releases/download/2.25-r0/glibc-2.25-r0.apk" -o /tmp/glibc.apk \
    && curl -L "https://github.com/andyshinn/alpine-pkg-glibc/releases/download/2.25-r0/glibc-bin-2.25-r0.apk" -o /tmp/glibc-bin.apk \
    && curl -L "https://github.com/andyshinn/alpine-pkg-glibc/releases/download/2.25-r0/glibc-i18n-2.25-r0.apk" -o /tmp/glibc-i18n.apk \
    && apk add --allow-untrusted /tmp/glibc*.apk \
    && /usr/glibc-compat/sbin/ldconfig /lib /usr/glibc-compat/lib \
    && /usr/glibc-compat/bin/localedef -i en_US -f UTF-8 en_US.UTF-8 \
    && rm -rf /tmp/glibc*apk /var/cache/apk/* \
    && mkdir -p $CONDA_DIR \
    && echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh \
    && curl https://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-x86_64.sh  -o mconda.sh \
    && /bin/bash mconda.sh -f -b -p $CONDA_DIR \
    && rm mconda.sh \
    && $CONDA_DIR/bin/conda install --yes conda==${CONDA_VER} \
    && conda update --all --yes \
    && conda config --set auto_update_conda False \
    && conda clean --all --yes \
    && pip --no-cache-dir install scikit-learn numpy scipy pandas statsmodels \
    && pip --no-cache-dir install git+https://github.com/yandexdataschool/modelgym.git \
    && pip --no-cache-dir install lightgbm \
    && apk add --update --no-cache \
    --virtual=.build-dependencies && \
    mkdir /src && \
    cd /src && \
    git clone --recursive https://github.com/dmlc/xgboost && \
    sed -i '/#define DMLC_LOG_STACK_TRACE 1/d' /src/xgboost/dmlc-core/include/dmlc/base.h && \
    sed -i '/#define DMLC_LOG_STACK_TRACE 1/d' /src/xgboost/rabit/include/dmlc/base.h && \
    apk del .build-dependencies && \
    apk add --update --no-cache \
    --virtual=.build-dependencies \
    make gfortran \
    python3-dev \
    py-setuptools g++ && \
    apk add --no-cache openblas lapack-dev libexecinfo-dev libstdc++ libgomp && \
    ln -s locale.h /usr/include/xlocale.h && \
    cd /src/xgboost; make -j4 && \
    cd /src/xgboost/python-package && \
    python3 setup.py install && \
    rm /usr/include/xlocale.h && \
    rm -r /root/.cache && \
    rm -rf /src && \
    apk del .build-dependencies

RUN /usr/glibc-compat/bin/localedef -i en_US -f UTF-8 en_US.UTF-8
RUN pip install azure-storage-file
RUN pip install google
RUN pip install grpcio
RUN pip install protobuf
RUN mkdir ~/repo-storage-worker/test -p
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    AFSSHARE="myshare" \
    WONDERCOMPUTECONFIG="/config.yaml" \
    REPO_STORAGE="~/repo-storage-worker"

COPY hyperoptWorker/*.py /hyperopt-worker/hyperoptWorker/
COPY setup.py /hyperopt-worker
COPY certs/* /certs/
COPY config.yaml /config.yaml
RUN pip --no-cache install /hyperopt-worker

CMD [ "python", "/hyperopt-worker/hyperoptWorker/hyperopt_worker.py"]
