FROM pytorch/pytorch

#Copying over requirement files
COPY requirements.txt .

RUN pip install --upgrade pip

#Expose port 1234 for notebook
EXPOSE 8888

#installing python requirements
RUN pip install -r requirements.txt
