FROM centos:7
RUN yum install -y https://repo.ius.io/ius-release-el7.rpm
RUN yum install -y python36u python36-pip python36u-devel

ENV HOST 0.0.0.0
ENV PORT 8080
ENV DEBUG true

ENV LC_ALL en_US.utf-8
ENV LANG en_US.etf-8

ENV CELERY_BACKEND redis://redis:6379/0
ENV CELERY_BROKER redis://redis:6379/0

COPY . /API
WORKDIR /API

EXPOSE 8080
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["api.py"]
