# %%
from model_xgb import model, predict
from user_input import input_diologue



def main():
    print("Your Movie Score:")
    new_df = input_diologue()
    result = predict(model, new_df)
    print("Estimated Movie Score:")
    print(result)

if __name__ == "__main__":
    main()
# %%