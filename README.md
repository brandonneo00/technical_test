# README

### For Task 2e:
- Ensure that you have Docker installed
- In your terminal, change directory to the asr folder (where the Dockerfile is located)
- Run `docker build -t asr-api .` in the terminal
- Run `docker run -d --name asr-api -p 8001:8001 asr-api` in the terminal
- To test the asr_api end point, you may use postman to test and attach a sample MP3 under `form-data` in Body, do not use `binary` to attach a sample MP3 in Body  


### Due to the size of the common_voice dataset being too large to push into Git, you may have to download the common_voice dataset on your own and the common_voice folder under a data folder in the main directory




