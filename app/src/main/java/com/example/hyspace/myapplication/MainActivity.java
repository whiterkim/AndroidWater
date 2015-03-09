package com.example.hyspace.myapplication;

import android.app.Activity;
import android.app.NativeActivity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;


public class MainActivity extends NativeActivity {
    static {
        System.loadLibrary("native-activity");
    }


}
