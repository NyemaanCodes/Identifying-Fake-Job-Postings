# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'DataSet.csv'
    import pandas as pd
    df = pd.read_csv(path, nrows=100,index_col=False)
    df.reset_index()
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

def preProcessData(request):
    from .utility.PreprocessedData import preProcessed_data_view
    data = preProcessed_data_view()
    return render(request, 'users/preproccessed_data.html', {'data': data})


def Model_Results(request):
    from users.utility import PreprocessedData
    nb_report = PreprocessedData.build_naive_bayes()
    knn_report = PreprocessedData.build_knn()
    dt_report = PreprocessedData.build_decsionTree()
    rf_report = PreprocessedData.build_randomForest()
    svm_report = PreprocessedData.build_svm()
    mlp_report = PreprocessedData.build_mlp()
    return render(request, 'users/ml_reports.html', {'nb': nb_report,"knn":knn_report, 'dt': dt_report, 'rf': rf_report, 'svm': svm_report,'mlp':mlp_report})

def user_input_prediction(request):
    if request.method=='POST':
        from .utility import PreprocessedData
        joninfo  = request.POST.get('joninfo')
        result = PreprocessedData.predict_userInput(joninfo)
        print(request)
        return render(request, 'users/testform.html', {'result': result})
    else:
        return render(request,'users/testform.html',{})