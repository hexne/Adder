#include <iostream>
#include <random>
#include <tuple>
#include <array>
#include <filesystem>
#include <fstream>
#include <filesystem>


constexpr int InputCount = 2;
constexpr int Nerou = 100;
constexpr int TrainMaxLimit = 100000;

std::filesystem::path config_file_path = "./config.txt";
double InputWeight[InputCount][Nerou];
double HindWeight[Nerou];
double HindValue[Nerou];
double Output;

double LearningRate;

struct Operand {
    double left,right,result;
};
std::array<Operand, 10000> Operands;



void ChangeRate(int count) {
    if (count < pow(10, 3))
        LearningRate = 0.05;
    else if (count < 3 * pow(10,3))        // 10 w
        LearningRate = 0.001;
    else if (count < 5 * pow(10,3))        // 10 w
        LearningRate = 0.0001;
    else
        LearningRate = 0.00001;
    /*if (count < pow(10, 3))
        LearningRate = 0.05;
    else if (count < pow(10,4))        // 10 w
        LearningRate = 0.005;
    else if (count < 2 * pow(10,4))       // 100w
        LearningRate = 0.0005;
    else if (count < 3 * pow(10,4))      // 1kw
        LearningRate = 0.00005;
    else if (count < 4 * pow(10,4))
        LearningRate = 0.000005;
    else if (count < 5 * pow(10,4))
        LearningRate = 0.0000005;
    else
        LearningRate = 0.000000001;*/
}

std::tuple<double, double, double> CreateRandom() {
	static std::random_device device;
	static std::mt19937 engin(device());
	static std::uniform_real_distribution<> dis(0, 1000);
	double left = dis(engin);
	double right = dis(engin);
	return { left, right, left + right };
}


template <typename T, size_t MaxSize>
class AverageQueue {
	std::array<T, MaxSize + 1> data_{};
	size_t max_size_ = MaxSize + 1;
	int front_{}, rear_{};
public:

	AverageQueue() = default;

	AverageQueue(const std::initializer_list<T> &init_list) {
		if (init_list.size() > max_size_)
			throw std::invalid_argument("AverageQueue is too many elements");

		front_ = 0;
		rear_ = init_list.size();
		std::copy(init_list.begin(), init_list.end(), data_.begin());
	}

	void push_back(const T &val) {
		int rear_back = rear_;
		data_[rear_++] = val;
		rear_ %= max_size_;

		if ((rear_back + 1) % max_size_ == front_) {
			front_++;
			front_ %= max_size_;
		}
	}
	T operator()() {
		int size = this->size();
		T ret{};
		for (int i = 0; i < size; ++i) {
			int pos = (front_ + i) % max_size_;
			ret += data_[pos];
		}
		return ret / size;
	}


	size_t size() const {
		int size = rear_ - front_;
		if (size > 0)
			return size;
		return size + max_size_;
	}

};


void Normalization(double& left, double& right, double& result) {
	left /= 1000;
	right /= 1000;
	result /= 1000;
}
void ReNormalization(double& left, double& right, double& result) {
	left *= 1000;
	right *= 1000;
	result *= 1000;
}

void InitNetwork() {
	std::random_device device;
	std::mt19937 engin(device());
	std::uniform_real_distribution<> dis(-1, 1);
	for (int i = 0; i < Nerou; ++i) {
		InputWeight[0][i] = dis(engin);
		InputWeight[1][i] = dis(engin);

		HindWeight[i] = dis(engin);
	}
}

void front(double left, double right) {
	for (int i = 0; i < Nerou; ++i) {
		double x = InputWeight[0][i] * left + InputWeight[1][i] * right;
		HindValue[i] = 1.0 / (1 + expf(-x));
	}

	Output = 0;
	for (int i = 0; i < Nerou; ++i)
		Output += HindValue[i] * HindWeight[i];
}

void back(double left, double right, double result) {
	static double d_hind[Nerou]{};
	for (int i = 0; i < Nerou; ++i) {
		d_hind[i] = (Output - result) * HindValue[i];
		HindWeight[i] -= LearningRate * d_hind[i];
	}
	static double d_input[InputCount][Nerou]{};
	for (int i = 0; i < Nerou; ++i) {
		d_input[0][i] = (Output - result) * HindWeight[i] * HindValue[i] * (1 - HindValue[i]) * left;
		d_input[1][i] = (Output - result) * HindWeight[i] * HindValue[i] * (1 - HindValue[i]) * right;
		InputWeight[0][i] -= LearningRate * d_input[0][i];
		InputWeight[1][i] -= LearningRate * d_input[1][i];
	}
}

void SaveConfig() {

    std::ofstream out(config_file_path);
    for (int i = 0;i < Nerou; ++i)
        out << InputWeight[0][i] << ' ' << InputWeight[1][i] << '\n';

    for (int i = 0;i < Nerou; ++i)
        out << HindWeight[i] << ' ';

}
void train() {
	int count{};
	double error_value{};
	AverageQueue<double, 10000> queue;

	do {
        count ++;
	    for (auto [left,right,result] : Operands) {
            ChangeRate(count);
            Normalization(left, right, result);
            front(left, right);

            error_value = fabs(Output - result) / result;
            queue.push_back(error_value);

            back(left, right, result);
            ReNormalization(left, right, Output);
	    }
	    SaveConfig();
	    std::cout << count << " , average is " << queue() << std::endl;
	} while (queue() > 0.000005);
}

void LoadConfig() {
    if (!std::filesystem::exists(config_file_path)) {
        InitNetwork();
        train();
        return;
    }
    std::ifstream in(config_file_path);
    for (int i = 0;i < Nerou; ++i)
        in >> InputWeight[0][i] >> InputWeight[1][i];

    for (int i = 0;i < Nerou; ++i)
        in >> HindValue[i];
}

double Inference(double left, double right) {
	front(left, right);
	return Output;
}

void CreateTestNumber() {
    for (auto &[left, right, result] : Operands) {
        auto [op1,op2,res] = CreateRandom();
        left = op1;
        right = op2;
        result = res;
    }
}


int main(int argc, char* argv[]) {
    LoadConfig();
    CreateTestNumber();
    SaveConfig();
	std::cout << Inference(1.5, 2.1) << std::endl;

	return 0;
}
