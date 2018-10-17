/*
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.View.OnTouchListener;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.Toast;

import com.vansuita.gaussianblur.GaussianBlur;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Vector;

import org.apache.http.params.CoreConnectionPNames;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.types.UInt8;

/**
 * Sample activity that stylizes the camera preview according to "A Learned Representation For
 * Artistic Style" (https://arxiv.org/abs/1610.07629)
 */
public class StylizeActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Copy these lines below
  private TensorFlowInferenceInterface inferenceInterface;

  private static final String MODEL_FILE = "file:///android_asset/stylize_quantized.pb";
  private static final String INPUT_NODE = "input";
  private static final String STYLE_NODE = "style_num";
  private static final String OUTPUT_NODE = "transformer/expand/conv3/conv/Sigmoid";

  private TensorFlowInferenceInterface inferenceInterfaceSegmenting;
  private static final String SEGMENT_MODEL_FILE = "file:///android_asset/frozen_inference_graph.pb";
  private static final String SEGMENT_INPUT_NODE = "ImageTensor";
  private static final String SEGMENT_OUTPUT_NODE = "SemanticPredictions";
  private float[] floatValuesSegmenting;

  private static final int NUM_STYLES = 26;

  private static final boolean SAVE_PREVIEW_BITMAP = false;

  // Whether to actively manipulate non-selected sliders so that sum of activations always appears
  // to be 1.0. The actual style input tensor will be normalized to sum to 1.0 regardless.
  private static final boolean NORMALIZE_SLIDERS = true;

  private static final float TEXT_SIZE_DIP = 12;

  private static final boolean DEBUG_MODEL = false;

  private static final int[] SIZES = {128, 192, 256, 384, 512, 720};

  private static final Size DESIRED_PREVIEW_SIZE = new Size(1280, 720);

  // Start at a medium size, but let the user step up through smaller sizes so they don't get
  // immediately stuck processing a large image.
  private int desiredSizeIndex = -1;
  private int desiredSize = 256;
  private int initializedSize = 0;

  private Integer sensorOrientation;

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private final float[] styleVals = new float[NUM_STYLES];
  private int[] intValues;
  private float[] floatValues;
  private float[] floatValuesNewImage;

  private int frameNum = 0;

  private Bitmap cropCopyBitmap;
  private Bitmap originalBitmap;

  private boolean computing = false;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private BorderedText borderedText;

  private long lastProcessingTimeMs;

  private int lastOtherStyle = 1;

  private boolean allZero = false;

  private ImageGridAdapter adapter;
  private GridView grid;

  private final OnTouchListener gridTouchAdapter =
      new OnTouchListener() {
        ImageSlider slider = null;

        @Override
        public boolean onTouch(final View v, final MotionEvent event) {
          switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
              for (int i = 0; i < NUM_STYLES; ++i) {
                final ImageSlider child = adapter.items[i];
                final Rect rect = new Rect();
                child.getHitRect(rect);
                if (rect.contains((int) event.getX(), (int) event.getY())) {
                  slider = child;
                  slider.setHilighted(true);
                }
              }
              break;

            case MotionEvent.ACTION_MOVE:
              if (slider != null) {
                final Rect rect = new Rect();
                slider.getHitRect(rect);

                final float newSliderVal =
                    (float)
                        Math.min(
                            1.0,
                            Math.max(
                                0.0, 1.0 - (event.getY() - slider.getTop()) / slider.getHeight()));

                setStyle(slider, newSliderVal);
              }
              break;

            case MotionEvent.ACTION_UP:
              if (slider != null) {
                slider.setHilighted(false);
                slider = null;
              }
              break;

            default: // fall out

          }
          return true;
        }
      };

  @Override
  public void onCreate(final Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_stylize;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  public static Bitmap getBitmapFromAsset(final Context context, final String filePath) {
    final AssetManager assetManager = context.getAssets();

    Bitmap bitmap = null;
    try {
      final InputStream inputStream = assetManager.open(filePath);
      bitmap = BitmapFactory.decodeStream(inputStream);
    } catch (final IOException e) {
      LOGGER.e("Error opening bitmap!", e);
    }

    return bitmap;
  }

  private class ImageSlider extends ImageView {
    private float value = 0.0f;
    private boolean hilighted = false;

    private final Paint boxPaint;
    private final Paint linePaint;

    public ImageSlider(final Context context) {
      super(context);
      value = 0.0f;

      boxPaint = new Paint();
      boxPaint.setColor(Color.BLACK);
      boxPaint.setAlpha(128);

      linePaint = new Paint();
      linePaint.setColor(Color.WHITE);
      linePaint.setStrokeWidth(10.0f);
      linePaint.setStyle(Style.STROKE);
    }

    @Override
    public void onDraw(final Canvas canvas) {
      super.onDraw(canvas);
      final float y = (1.0f - value) * canvas.getHeight();

      // If all sliders are zero, don't bother shading anything.
      if (!allZero) {
        canvas.drawRect(0, 0, canvas.getWidth(), y, boxPaint);
      }

      if (value > 0.0f) {
        canvas.drawLine(0, y, canvas.getWidth(), y, linePaint);
      }

      if (hilighted) {
        canvas.drawRect(0, 0, getWidth(), getHeight(), linePaint);
      }
    }

    @Override
    protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
      super.onMeasure(widthMeasureSpec, heightMeasureSpec);
      setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
    }

    public void setValue(final float value) {
      this.value = value;
      postInvalidate();
    }

    public void setHilighted(final boolean highlighted) {
      this.hilighted = highlighted;
      this.postInvalidate();
    }
  }

  private class ImageGridAdapter extends BaseAdapter {
    final ImageSlider[] items = new ImageSlider[NUM_STYLES];
    final ArrayList<Button> buttons = new ArrayList<>();

    {
      final Button sizeButton =
          new Button(StylizeActivity.this) {
            @Override
            protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
              super.onMeasure(widthMeasureSpec, heightMeasureSpec);
              setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
            }
          };
      sizeButton.setText("" + desiredSize);
      sizeButton.setOnClickListener(
          new OnClickListener() {
            @Override
            public void onClick(final View v) {
              desiredSizeIndex = (desiredSizeIndex + 1) % SIZES.length;
              desiredSize = SIZES[desiredSizeIndex];
              sizeButton.setText("" + desiredSize);
              sizeButton.postInvalidate();
            }
          });

      final Button saveButton =
          new Button(StylizeActivity.this) {
            @Override
            protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
              super.onMeasure(widthMeasureSpec, heightMeasureSpec);
              setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
            }
          };
      saveButton.setText("save");
      saveButton.setTextSize(12);

      saveButton.setOnClickListener(
          new OnClickListener() {
            @Override
            public void onClick(final View v) {
              if (cropCopyBitmap != null) {
                // make a copy of cropCopyBitmap so none of the new bitmaps are affected by
                // changing camera views
                originalBitmap = Bitmap.createBitmap(cropCopyBitmap);

                // stylize image will stylize the image, segment the image, and apply
                // lumanince matching
                stylizeImage(originalBitmap);

                // notify image saved
                Toast.makeText(
                        StylizeActivity.this,
                        "Saved image to: /sdcard/tensorflow/" + "stylized" + frameNum + ".png",
                        Toast.LENGTH_LONG)
                    .show();
              }
            }
          });

      buttons.add(sizeButton);
      buttons.add(saveButton);

      for (int i = 0; i < NUM_STYLES; ++i) {
        LOGGER.v("Creating item %d", i);

        if (items[i] == null) {
          final ImageSlider slider = new ImageSlider(StylizeActivity.this);
          final Bitmap bm =
              getBitmapFromAsset(StylizeActivity.this, "thumbnails/style" + i + ".jpg");
          slider.setImageBitmap(bm);

          items[i] = slider;
        }
      }

    }

    @Override
    public int getCount() {
      return buttons.size() + NUM_STYLES;
    }

    @Override
    public Object getItem(final int position) {
      if (position < buttons.size()) {
        return buttons.get(position);
      } else {
        return items[position - buttons.size()];
      }
    }

    @Override
    public long getItemId(final int position) {
      return getItem(position).hashCode();
    }

    @Override
    public View getView(final int position, final View convertView, final ViewGroup parent) {
      if (convertView != null) {
        return convertView;
      }
      return (View) getItem(position);
    }
  }

  // takes in stylized image
  private void addToTextureCopyLumanince(final Bitmap bitmap, final Bitmap backupOriginalBitmap) {
    // [R,G,B, R,G,B, ...] of the original image
    // floatValues

    // TODO: (chrismgeorge) Remove global lists/arrays. They are messing with my head.

    // intValues is now for the stylized image
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // get the float values of old new pixel
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      floatValuesNewImage[i * 3] = ((val >> 16) & 0xFF);
      floatValuesNewImage[i * 3 + 1] = ((val >> 8) & 0xFF);
      floatValuesNewImage[i * 3 + 2] = (val & 0xFF);
    }

    // get info for original image
    backupOriginalBitmap.getPixels(intValues, 0, backupOriginalBitmap.getWidth(),
                          0, 0, backupOriginalBitmap.getWidth(), backupOriginalBitmap.getHeight());

    // get the float values of the new pixel
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      floatValues[i * 3] = ((val >> 16) & 0xFF);
      floatValues[i * 3 + 1] = ((val >> 8) & 0xFF);
      floatValues[i * 3 + 2] = (val & 0xFF);
    }

    for (int i = 0; i < floatValues.length; i+=3){
      float[] newPixel = {floatValuesNewImage[i], floatValuesNewImage[i+1], floatValuesNewImage[i+2]};
      double newLuminance = (0.2126*newPixel[0] + 0.7152*newPixel[1] + 0.0722*newPixel[2]);

      float[] originalPixel = {floatValues[i], floatValues[i+1], floatValues[i+2]};
      double originalLuminance = (0.2126*originalPixel[0] + 0.7152*originalPixel[1] + 0.0722*originalPixel[2]);

      // a number < 1 means that old pixel is brighter than the new one
      // so we need to multiple the new pixel by the ratio to brighten it
      double ratio = (originalLuminance / newLuminance);

      if (ratio > 1) {
        ratio = 1 + Math.log10(ratio);
      } else {
        // scale to .8 -> 1
        ratio = (ratio/5.0f) + .8f;
      }

      double[] brightPixel = {newPixel[0]*ratio, newPixel[1]*ratio, newPixel[2]*ratio};
      // keep the values between 0, 255
      List<Double> doubleList = Arrays.asList(brightPixel[0], brightPixel[1], brightPixel[2]);
      double maxNewValue = Collections.max(doubleList);

      if (maxNewValue > 255.0f) {
        double diff = maxNewValue / 255.0;
        floatValues[i] = (float) (brightPixel[0] / diff / 255.0f);
        floatValues[i+1] = (float) (brightPixel[1] / diff / 255.0f);
        floatValues[i+2] = (float) (brightPixel[2] / diff / 255.0f);
      } else {
        floatValues[i] = (float) (brightPixel[0] / 255.0f);
        floatValues[i+1] = (float) (brightPixel[1] / 255.0f);
        floatValues[i+2] = (float) (brightPixel[2] / 255.0f);
      }
    }

    for (int i = 0; i < intValues.length; ++i) {
      intValues[i] =
              0xFF000000
                      | (((int) (floatValues[i * 3] * 255)) << 16)
                      | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
                      | ((int) (floatValues[i * 3 + 2] * 255));
    }

    bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
  }



  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    inferenceInterfaceSegmenting = new TensorFlowInferenceInterface(getAssets(), SEGMENT_MODEL_FILE);

    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            renderDebug(canvas);
          }
        });

    adapter = new ImageGridAdapter();
    grid = (GridView) findViewById(R.id.grid_layout);
    grid.setAdapter(adapter);
    grid.setOnTouchListener(gridTouchAdapter);

    setStyle(adapter.items[0], 1.0f);
  }

  private void setStyle(final ImageSlider slider, final float value) {
    slider.setValue(value);

    if (NORMALIZE_SLIDERS) {
      // Slider vals correspond directly to the input tensor vals, and normalization is visually
      // maintained by remanipulating non-selected sliders.
      float otherSum = 0.0f;

      for (int i = 0; i < NUM_STYLES; ++i) {
        if (adapter.items[i] != slider) {
          otherSum += adapter.items[i].value;
        }
      }

      if (otherSum > 0.0) {
        float highestOtherVal = 0;
        final float factor = otherSum > 0.0f ? (1.0f - value) / otherSum : 0.0f;
        for (int i = 0; i < NUM_STYLES; ++i) {
          final ImageSlider child = adapter.items[i];
          if (child == slider) {
            continue;
          }
          final float newVal = child.value * factor;
          child.setValue(newVal > 0.01f ? newVal : 0.0f);

          if (child.value > highestOtherVal) {
            lastOtherStyle = i;
            highestOtherVal = child.value;
          }
        }
      } else {
        // Everything else is 0, so just pick a suitable slider to push up when the
        // selected one goes down.
        if (adapter.items[lastOtherStyle] == slider) {
          lastOtherStyle = (lastOtherStyle + 1) % NUM_STYLES;
        }
        adapter.items[lastOtherStyle].setValue(1.0f - value);
      }
    }

    final boolean lastAllZero = allZero;
    float sum = 0.0f;
    for (int i = 0; i < NUM_STYLES; ++i) {
      sum += adapter.items[i].value;
    }
    allZero = sum == 0.0f;

    // Now update the values used for the input tensor. If nothing is set, mix in everything
    // equally. Otherwise everything is normalized to sum to 1.0.
    for (int i = 0; i < NUM_STYLES; ++i) {
      styleVals[i] = allZero ? 1.0f / NUM_STYLES : adapter.items[i].value / sum;

      if (lastAllZero != allZero) {
        adapter.items[i].postInvalidate();
      }
    }
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;

    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (computing) {
        image.close();
        return;
      }

      if (desiredSize != initializedSize) {
        LOGGER.i(
            "Initializing at size preview size %dx%d, stylize size %d",
            previewWidth, previewHeight, desiredSize);
        rgbBytes = new int[previewWidth * previewHeight];
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(desiredSize, desiredSize, Config.ARGB_8888);

        frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                desiredSize, desiredSize,
                sensorOrientation, true);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        yuvBytes = new byte[3][];

        intValues = new int[desiredSize * desiredSize];
        floatValues = new float[desiredSize * desiredSize * 3];
        floatValuesNewImage = new float[desiredSize * desiredSize * 3];
        floatValuesSegmenting = new float[desiredSize * desiredSize * 3];
        initializedSize = desiredSize;
      }

      // keep
      computing = true;

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      ImageUtils.convertYUV420ToARGB8888(
          yuvBytes[0],
          yuvBytes[1],
          yuvBytes[2],
          previewWidth,
          previewHeight,
          yRowStride,
          uvRowStride,
          uvPixelStride,
          rgbBytes);

      image.close();
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
    requestRender();

    // keep
    computing = false;

    Trace.endSection();
  }

  private void stylizeImage(final Bitmap bitmap) {
    ++frameNum;
    String r = Integer.toString((int) (Math.random() * 1000000000));
    Bitmap backupOriginal = Bitmap.createBitmap(bitmap);


    // intvalues are the pixels of the original image, need tro translate them because
    // they come out as a big integer I think
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    if (DEBUG_MODEL) {
      // Create a white square that steps through a black background 1 pixel per frame.
      final int centerX = (frameNum + bitmap.getWidth() / 2) % bitmap.getWidth();
      final int centerY = bitmap.getHeight() / 2;
      final int squareSize = 10;
      for (int i = 0; i < intValues.length; ++i) {
        final int x = i % bitmap.getWidth();
        final int y = i / bitmap.getHeight();
        final float val =
                Math.abs(x - centerX) < squareSize && Math.abs(y - centerY) < squareSize ? 1.0f : 0.0f;
        floatValues[i * 3] = val;
        floatValues[i * 3 + 1] = val;
        floatValues[i * 3 + 2] = val;
      }
    }

    else {
      for (int i = 0; i < intValues.length; ++i) {
        final int val = intValues[i];
        floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
        floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
        floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
      }
    }

    // for segmentation
    byte[] mFlatIntValues = new byte[desiredSize*desiredSize*3];
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      mFlatIntValues[i * 3 + 0] = (byte)((val >> 16) & 0xFF);
      mFlatIntValues[i * 3 + 1] = (byte)((val >> 8) & 0xFF);
      mFlatIntValues[i * 3 + 2] = (byte)(val & 0xFF);
    }

    //// Segmenting of image
    int[] mOutputs = new int[desiredSize*desiredSize];

    inferenceInterfaceSegmenting.feed(SEGMENT_INPUT_NODE, mFlatIntValues, 1, bitmap.getWidth(), bitmap.getHeight(), 3);
    inferenceInterfaceSegmenting.run(new String[] {SEGMENT_OUTPUT_NODE}, isDebug());
    // important that mOutputs holds the data
    inferenceInterfaceSegmenting.fetch(SEGMENT_OUTPUT_NODE, mOutputs);

    // generate the segmented image for reference
    int w = desiredSize;
    int h = desiredSize;
    Bitmap output = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        output.setPixel(x, y, mOutputs[y * w + x] == 0 ? Color.WHITE : Color.BLACK);
      }
    }

    // save segmented image
    ImageUtils.saveBitmap(output, "segmented" + frameNum + r + ".png");

    // Copy the input data into TensorFlow.
    inferenceInterface.feed(INPUT_NODE, floatValues,
            1, bitmap.getWidth(), bitmap.getHeight(), 3);
    inferenceInterface.feed(STYLE_NODE, styleVals, NUM_STYLES);
    // Execute the output node's dependency sub-graph.
    inferenceInterface.run(new String[] {OUTPUT_NODE}, isDebug());
    // Copy the data from TensorFlow back into our array.
    inferenceInterface.fetch(OUTPUT_NODE, floatValues);

    // translate outputs of the inference to be readable by bitmaps
    for (int i = 0; i < intValues.length; ++i) {
      intValues[i] =
          0xFF000000
              | (((int) (floatValues[i * 3] * 255)) << 16)
              | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
              | ((int) (floatValues[i * 3 + 2] * 255));
    }

    // set pixels for pure stylized image
    bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    ImageUtils.saveBitmap(bitmap, "style_pic" + frameNum + r + ".png");

    // Add lumanice matching to the bitmap image before blurring it
    Bitmap copy = Bitmap.createBitmap(bitmap);
    addToTextureCopyLumanince(copy, backupOriginal);

    // Save lumanince matched image
    ImageUtils.saveBitmap(copy, "lumanice_stylized" + frameNum + r + ".png");

    // Apply gaussian blur to image
    Bitmap blurredBitmap = GaussianBlur.with(getBaseContext()).radius(10).render(copy);
    blurredBitmap = Bitmap.createScaledBitmap(blurredBitmap, desiredSize, desiredSize, false);

    // save blurred image for reference
    ImageUtils.saveBitmap(blurredBitmap, "blurred" + frameNum + r + ".png");

    // make a new bitmap, where the foreground is the unblurred original image,
    // and the background is the blurred image
    Bitmap blurredBothSameStyle = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
    Bitmap blurredDiffStyle = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        // background
        if (mOutputs[y * w + x] == 0){
          blurredBothSameStyle.setPixel(x, y, blurredBitmap.getPixel(x, y));

          // no filter foreground
          blurredDiffStyle.setPixel(x, y, blurredBitmap.getPixel(x, y));
        }
        // foreground
        else {
          blurredBothSameStyle.setPixel(x, y, copy.getPixel(x, y));

          // no filter foreground
          blurredDiffStyle.setPixel(x, y, backupOriginal.getPixel(x, y));
        }
      }
    }

    // save the new mix of an image
    ImageUtils.saveBitmap(blurredBothSameStyle, "blurred_two" + frameNum + r + ".png");

    // mix in reality
    ImageUtils.saveBitmap(blurredDiffStyle, "blurred_three" + frameNum + r + ".png");


  }

  private void renderDebug(final Canvas canvas) {
    final Bitmap texture = cropCopyBitmap;
    if (texture != null) {
      final Matrix matrix = new Matrix();
      final float scaleFactor =
          DEBUG_MODEL
              ? 4.0f
              : Math.min(
                  (float) canvas.getWidth() / texture.getWidth(),
                  (float) canvas.getHeight() / texture.getHeight());
      matrix.postScale(scaleFactor, scaleFactor);
      canvas.drawBitmap(texture, matrix, new Paint());
    }

    if (!isDebug()) {
      return;
    }

    final Bitmap copy = cropCopyBitmap;
    if (copy == null) {
      return;
    }

    canvas.drawColor(0x55000000);

    final Matrix matrix = new Matrix();
    final float scaleFactor = 2;
    matrix.postScale(scaleFactor, scaleFactor);
    matrix.postTranslate(
        canvas.getWidth() - copy.getWidth() * scaleFactor,
        canvas.getHeight() - copy.getHeight() * scaleFactor);
    canvas.drawBitmap(copy, matrix, new Paint());

    final Vector<String> lines = new Vector<>();

    // Add these three lines right here:
    final String[] statLines = inferenceInterface.getStatString().split("\n");
    Collections.addAll(lines, statLines);
    lines.add("");

    lines.add("Frame: " + previewWidth + "x" + previewHeight);
    lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
    lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
    lines.add("Rotation: " + sensorOrientation);
    lines.add("Inference time: " + lastProcessingTimeMs + "ms");
    lines.add("Desired size: " + desiredSize);
    lines.add("Initialized size: " + initializedSize);

    borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
  }
}
