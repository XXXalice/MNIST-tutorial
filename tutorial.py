
import tensorflow as tf
import input_data

#MNISTデータ読み込み
mnist = input_data.read_data_sets("data/", one_hot=True)

#画像データをxとする
x = tf.placeholder("float", [None,784])

#モデルの重みをwと設定する
w = tf.Variable(tf.zeros([784,10]))

#モデルのバイアス
b = tf.Variable(tf.zeros([10]))

#トレーニングデータ（x）とモデルの重み（w）の乗算後、モデルのバイアス（b）を足しソフトマックス関数を適応させる
y = tf.nn.softmax(tf.matmul(x,w) + b)

#正解のデータ
y_ = tf.placeholder("float",[None,10])

#損失関数をクロスエントロピーとする
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#学習係数を0.01として勾配降下アルゴリズムを用いてクロスエントロピーを最小化する
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#変数の初期化
init = tf.initialize_all_variables()

#セッションの作成
sess = tf.Session()

#セッションの開始及び初期化
sess.run(init)

for i in range(1000):

    #トレーニングデータからランダムに100個抽出する
    batch_xs, batch_ys = mnist.train.next_batch(100)

    #確率的勾配降下によりクロスエントロピーを最小化するような重みを更新
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

#予測値と正解地を比較してbool値にする
#argmax(y,1)は、予測値の各行で最大となるインデックスをひとつ返す
correct_prediction= tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#bool値を0もしくは1に変換して平均値をとる、これを正解率とする
accurary= tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accurary, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))