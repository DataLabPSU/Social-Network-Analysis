from django.shortcuts import render, redirect, render_to_response, get_object_or_404
from django.template import RequestContext
from .models import Post, Comment, Share, Profile
from . import forms
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
import datetime
import os


def home(request):
    '''
        os.chdir('/Users/adriankato/Documents/github/djangonetwork/static/videos')
        for i in os.listdir():
            user = User.objects.get(username='vic')
            temp = Post.objects.create(author=user)
            temp.title = 'Video'
            temp.text = 'This is a video'
            temp.videoname = i
            temp.save()
    '''

    #if datetime.datetime.now().hour > 12:
    #    return render(request,'twitterclone/end.html')
    d = {}
    followerscount = {}
    for user in User.objects.all():
        temp = user.profile.credibilityscore * (user.profile.real+1)/(user.profile.real+user.profile.fake+2)
        temp2 = 1 - user.profile.credibilityscore
        temp3 = 1 - (user.profile.real+1)/(user.profile.real+user.profile.fake+2)
        followeescredit = len(set(user.profile.following.split(" ")[1:])) / len(User.objects.all())
        d[user.username] = temp/(temp2*temp3) - followeescredit
        for i in set(user.profile.following.split(" ")[1:]):
            try:
                followerscount[i] += 1
            except:
                followerscount[i] = 1
    for username in d.keys():
        try:
            followers = followerscount[username]
        except:
            followers = 0
        print(d[username],username)
        d[username] = d[username] - followers/len(User.objects.all())
        temp = User.objects.get(username=username)
        temp.profile.credibilityscore = d[username]
        temp.save()


    if request.user.is_authenticated:
        if request.method == 'POST':
            try:
                follow = request.POST['follows']
                user = request.user
                user.profile.following = user.profile.following + " " + follow
                user.save()
            except:
                pass

            try:
                postid = request.POST['share']
                post = Post.objects.get(id=postid)
                if request.user != post.author:
                    newshare = Share.objects.create(shared=request.user)
                    newshare.postid = postid
                    newshare.save()
                    user = request.user
                    if post.real == 0:
                        user.profile.fake = user.profile.fake + 1
                    else:
                        user.profile.real = user.profile.real + 1
                    user.save()
                    for user in User.objects.all():
                        if request.user.username in user.profile.following.split(" "):
                            user.profile.notifications = user.profile.notifications + 1
                            user.profile.notificationsString += "Post shared by " + str(request.user) + " at " + str(datetime.datetime.now()) + "|"
                            user.save()

            except Exception as e:
                pass

            try:

                user = User.objects.get(pk=request.user.id)
                temp = user.profile.following.split(" ")
                temp.remove(request.POST['follow'])
                user.profile.following = ' '.join(temp)
                user.save()
            except:
                pass
            try:
                user = User.objects.get(pk=request.user.id)
                post = Post.objects.get(pk=request.POST['unique'])
                if post.author != user and request.POST['unique'] not in user.profile.liked:
                    post.likes += 1
                    post.save()
                    if post.real == 0:
                        user.profile.fake = user.profile.fake + 1
                    else:
                        user.profile.real = user.profile.real + 1
                    user.profile.liked = user.profile.liked + " " + request.POST['unique']
                    user.save()

                elif post.author != user and request.POST['unique'] in user.profile.liked:
                    temp = user.profile.liked.split(" ")
                    temp.remove(request.POST['unique'])
                    user.profile.liked = ' '.join(temp)
                    post.likes -= 1
                    if post.real == 0:
                        user.profile.fake = user.profile.fake - 1
                    else:
                        user.profile.real = user.profile.real - 1
                    post.save()
                    user.save()
            except:
                pass
            try:
                comment = request.POST['placeholder']
                newcomment = Comment.objects.create(author=request.user)
                newcomment.post = Post.objects.get(pk=request.POST['submit'])
                newcomment.text = comment
                newcomment.save()
                for user in User.objects.all():
                    if request.user.username in user.profile.following.split(" "):
                        user.profile.notifications = user.profile.notifications + 1
                        user.save()
            except:
                pass
        user = User.objects.get(pk=request.user.id)

        posts = Post.objects.filter(author=request.user)
        postlist = list(posts)
        shares = Share.objects.filter(shared=request.user)
        for z in shares:
            content = Post.objects.get(id=z.postid)
            content.author.username = content.author.username + ' Retweeted by ' + z.shared.username
            content.created_date = z.date
            postlist.append(content)
        comments = Comment.objects.all()
        if user.profile.following != "":
            for i in set(user.profile.following.split(" ")):
                if i != "":
                    print(i)
                    following = User.objects.get(username=i)
                    otherposts = Post.objects.filter(author=following)
                    posts = posts | otherposts
                    postlist.extend(list(otherposts))
                    sharedposts = Share.objects.filter(shared=following)
                    for z in sharedposts:
                        content = Post.objects.get(id=z.postid)
                        content.author.username = content.author.username + ' Shared by ' + z.shared.username
                        content.created_date = z.date
                        postlist.append(content)
        notificationsString = request.user.profile.notificationsString.split("|")
        userlist = User.objects.exclude(pk=request.user.id)
        finaloutput = []
        temp = (Profile.objects.all().order_by('-credibilityscore'))[:10]

        for i in temp:
            if i.user != request.user:
                numfollowers = len(set(i.following.split(" ")[1:]))
                finaloutput.append([i.user, numfollowers])
        following = []
        for i in set(user.profile.following.split(" ")[1:]):
            following.append(User.objects.get(username=i))
        context = {
            'posts': postlist,
            'comments': comments,
            'FOLLOWING': following,
            'test': (user.profile.following.split(" ")),
            'currentuser': request.user,
            'notifications': notificationsString,
            'users': userlist,
            'numfollowers': finaloutput,
        }

        return render(request, "twitterclone/home.html", context)
    else:
        return redirect('login')


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'twitterclone/signup.html', {'form': form})


def addpost(request):
    if request.method == 'POST':
        form = forms.PostForm(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.author = request.user
            instance.save()
            return redirect('home')
    else:
        form = forms.PostForm()

    return render(request, 'twitterclone/create.html', {'form': form})


def notification(request):
    temp = request.user.profile.notificationsString
    num = request.user.profile.notifications
    context = {
        'notifications': temp.split("|"),
        'num': num,
    }
    return render(request, 'twitterclone/notification.html', context)


def testfollow(request):
    if request.method == 'POST':
        user = User.objects.get(pk=request.user.id)
        user.profile.following = user.profile.following + " " + request.POST['follow']
        user.save()

    userlist = User.objects.exclude(pk=request.user.id)
    context = {
        'users': userlist
    }
    return render(request, 'twitterclone/follow.html', context)

def pick(request):
    if request.method == 'POST':
        user = User.objects.get(pk=request.user.id)
        user.profile.imagename = request.POST['hidden']
        user.save()
    context = {
        'images':['image'+str(i) for i in range(4)],
        'user':request.user,
    }
    print(context)
    return render(request,'twitterclone/profile.html',context)
