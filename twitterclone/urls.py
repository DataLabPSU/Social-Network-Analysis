from django.conf.urls import url
from . import views
from django.contrib import admin
from django.urls import include
from django.urls import path
urlpatterns=[
path('signup/',views.signup,name='signup'),
path('',views.home,name='home'),
path('create/',views.addpost,name='addpost'),
path('notification/', views.notification, name='notification'),
path('settings/',views.pick,name='settings'),
path('pm/',views.pm,name='pm'),
path('instructions/',views.instructions,name='instructions'),
path('final/',views.survey,name='survey'),
url(r'^ajax/updatelike/$', views.updatelike, name='updatelike'),
url(r'^ajax/sharepost/$', views.sharepost, name='sharepost'),

# dumps data
path('processdata/', views.processdata,name='processdata'),

# deletes all posts excluding videos
path('deleteposts/', views.deleteposts,name='deleteposts'),

# adds new videos
path('addvideos/', views.addvideos,name='addvideos'),

# deletes all users except admin 
path('deleteusers/', views.deleteusers,name='deleteusers'),

# resets user information
path('resetuserdata/', views.resetuserdata,name='resetuserdata'),

# resets video info
path('resetvideos/', views.resetvideos,name='resetvideos'),

# adds youtube video ids to existing videos
path('addvideoids/', views.addvideoids, name='addvideoids'),

# update credibility score
path('updatecred/', views.updateCredibilityScore, name='updatecred'),

#loadtesting
#path('createusers/', views.createusers,name='createusers'),
#path('loadtest/', views.loadtest,name='loadtest')
# path('follow/',views.testfollow,name='testfollow')
]



